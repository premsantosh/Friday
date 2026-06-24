"""
Phone channel — autonomous calling via Bland.ai (M5).

Unlike the browser channels, phone is **asynchronous**: after the user approves
the call plan, we place the call and the session goes to WAITING; the workflow's
on_tick() (driven by the MT3 background runner) polls Bland for the structured
outcome. Off-hours calls are deferred to the business's next opening.

`dial_and_bridge` is a non-autonomous fallback: it hands the number to the user
to call themselves. Google Voice is not supported (no call API).

The Bland HTTP client is injectable so the workflow is testable without network.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.harness import Sink, SinkMode, guard

from ..models import ChannelDecision, ReservationMethod
from .base import Availability, AvailabilityStatus, BookingResult, CommitPlan, ReservationChannel

logger = logging.getLogger(__name__)

# What may leave for Bland: the destination, the call plan, minimal request
# context, and the outcome schema. The free text is still scanned (no card
# data is ever voiced; spec §8 L6).
BLAND_SINK = Sink("bland", SinkMode.ALLOWLIST,
                  frozenset({"phone_number", "task", "request_data", "analysis_schema"}))

POLL_SECONDS = 60          # how often to poll Bland for a result
RETRY_DELAY_SECONDS = 300  # wait between call attempts on no-answer


class PhoneOutcome(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    NO_AVAILABILITY = "no_availability"
    EMAIL_REQUESTED = "email_requested"
    CALLBACK_REQUIRED = "callback_required"
    NEEDS_INFO = "needs_info"
    FAILED = "failed"          # no answer / busy / voicemail


@dataclass
class CallResult:
    outcome: PhoneOutcome
    message: str = ""
    confirmation: Optional[str] = None
    negotiated_time: Optional[str] = None
    email: Optional[str] = None         # captured if they asked to be emailed
    question: Optional[str] = None       # captured for needs_info


class BlandClient:
    """Minimal Bland.ai calls client. Injectable poster/getter for tests."""

    def __init__(self, api_key: str, base_url: str = "https://api.bland.ai/v1",
                 poster: Optional[Callable] = None, getter: Optional[Callable] = None):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._poster = poster
        self._getter = getter

    @classmethod
    def from_env(cls) -> Optional["BlandClient"]:
        key = os.getenv("BLAND_API_KEY")
        return cls(key) if key else None

    def place_call(self, phone_number: str, task: str,
                   request_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        payload = {
            "phone_number": phone_number,
            "task": task,
            "request_data": request_data or {},
            # Ask Bland to return a structured outcome under "analysis".
            "analysis_schema": {
                "outcome": "one of: confirmed, no_availability, email_requested, "
                           "callback_required, needs_info, failed",
                "confirmation_number": "string or null",
                "negotiated_time": "string or null",
                "email_address": "string or null",
                "question": "string or null",
            },
        }
        # Outside the try: a blocked payload must fail the call loudly (the
        # gate turns it into an honest refusal), not look like a network blip.
        guard(BLAND_SINK, payload)
        try:
            poster = self._poster
            if poster is None:
                import requests
                poster = requests.post
            resp = poster(
                f"{self.base_url}/calls",
                headers={"authorization": self.api_key},
                json=payload,
                timeout=15,
            )
            data = resp.json()
            return data.get("call_id") or data.get("id")
        except Exception:
            logger.warning("Bland place_call failed.", exc_info=True)
            return None

    def get_result(self, call_id: str) -> Optional[Dict[str, Any]]:
        try:
            getter = self._getter
            if getter is None:
                import requests
                getter = requests.get
            resp = getter(
                f"{self.base_url}/calls/{call_id}",
                headers={"authorization": self.api_key},
                timeout=15,
            )
            return resp.json()
        except Exception:
            logger.warning("Bland get_result failed.", exc_info=True)
            return None


class PhoneChannel(ReservationChannel):
    method = ReservationMethod.PHONE
    can_commit = True
    is_async = True

    def __init__(self, client: Optional[BlandClient] = None,
                 callback_number: str = "", guest_name: str = ""):
        self.client = client
        self.callback_number = callback_number
        self.guest_name = guest_name

    # ----- ReservationChannel surface (commit is unused; workflow runs the async flow)
    async def check_availability(self, slots: Dict[str, Any]) -> Availability:
        return Availability(status=AvailabilityStatus.UNKNOWN,
                            note="I'll ask them on the call.")

    async def prepare(self, slots: Dict[str, Any], decision: ChannelDecision) -> CommitPlan:
        return CommitPlan(
            channel="phone",
            summary=f"Call {decision.business_name or slots.get('business_name')} to book",
            details={
                "business_name": decision.business_name or slots.get("business_name"),
                "phone": decision.phone,
                "date": slots.get("date"),
                "time": slots.get("time"),
                "party_size": slots.get("party_size"),
                "special_requests": slots.get("special_requests"),
                "hours": decision.hours,
                "timezone": decision.timezone,
                "call_plan": self.build_call_plan(slots, decision),
            },
            requires_card=False,
        )

    async def commit(self, plan: CommitPlan, payment: Any = None) -> BookingResult:
        # Phone is async; the workflow drives place_call/poll directly, not commit().
        return BookingResult(success=False, needs_manual=True,
                             message="Phone bookings are placed asynchronously.",
                             error="use_async_flow")

    # ----- call plan + Bland lifecycle
    def build_call_plan(self, slots: Dict[str, Any], decision: ChannelDecision) -> str:
        biz = decision.business_name or slots.get("business_name")
        party = slots.get("party_size")
        date = slots.get("date")
        time = slots.get("time")
        name = self.guest_name or slots.get("guest_name") or "the caller"
        extras = slots.get("special_requests")
        lines = [
            f"You are a polite assistant calling {biz} to make a reservation on behalf of {name}.",
            f"Request a table for {party} on {date} at {time}.",
            "If that exact time is unavailable, ask about nearby times the same day and accept "
            "the closest one.",
            f"If they ask for a callback number, give {self.callback_number or 'the number on file'}.",
            "If they ask you to email a request instead, get the email address.",
            "Be concise and courteous; confirm the final time and any confirmation number.",
        ]
        if extras:
            lines.insert(2, f"Note this request: {extras}.")
        return " ".join(lines)

    def place_call(self, plan: CommitPlan) -> Optional[str]:
        if self.client is None:
            return None
        phone = plan.details.get("phone")
        if not phone:
            return None
        return self.client.place_call(
            phone, plan.details.get("call_plan", ""),
            request_data={"business": plan.details.get("business_name")},
        )

    def poll(self, call_id: str) -> CallResult:
        if self.client is None:
            return CallResult(PhoneOutcome.FAILED, message="No phone client configured.")
        data = self.client.get_result(call_id)
        if not data:
            return CallResult(PhoneOutcome.PENDING)
        return self._classify(data)

    @staticmethod
    def _classify(data: Dict[str, Any]) -> CallResult:
        # Still in progress?
        status = (data.get("status") or data.get("queue_status") or "").lower()
        completed = data.get("completed")
        if completed is False or status in ("queued", "in-progress", "ringing", "started"):
            return CallResult(PhoneOutcome.PENDING)

        analysis = data.get("analysis") or {}
        outcome_str = (analysis.get("outcome") or "").lower().strip()
        text = f"{data.get('summary', '')} {data.get('concatenated_transcript', '')}".lower()

        def has(*words):
            return any(w in text for w in words)

        if outcome_str in PhoneOutcome._value2member_map_:
            outcome = PhoneOutcome(outcome_str)
        elif has("no availability", "fully booked", "nothing available", "no tables", "can't accommodate"):
            # Check unavailability first — "fully booked" contains "booked".
            outcome = PhoneOutcome.NO_AVAILABILITY
        elif has("confirmed", "all set", "see you", "you're booked", "reservation is set"):
            outcome = PhoneOutcome.CONFIRMED
        elif has("voicemail", "no answer", "didn't answer", "busy"):
            # Before the email check — "voicemail" contains "email".
            outcome = PhoneOutcome.FAILED
        elif has("email"):
            outcome = PhoneOutcome.EMAIL_REQUESTED
        elif has("call back", "callback", "call you back"):
            outcome = PhoneOutcome.CALLBACK_REQUIRED
        else:
            outcome = PhoneOutcome.NEEDS_INFO

        return CallResult(
            outcome=outcome,
            message=data.get("summary", ""),
            confirmation=analysis.get("confirmation_number") or None,
            negotiated_time=analysis.get("negotiated_time") or None,
            email=analysis.get("email_address") or None,
            question=analysis.get("question") or None,
        )

    # ----- business hours / off-hours deferral
    def is_open_now(self, details: Dict[str, Any],
                    now: Optional[datetime] = None) -> Tuple[bool, Optional[float]]:
        """Return (open_now, next_open_epoch). If hours are unknown, assume open now."""
        hours = details.get("hours")
        now = now or datetime.now()
        intervals = _parse_hours(hours)
        if not intervals:
            return True, None  # unknown hours → don't defer

        cur_min = now.hour * 60 + now.minute
        for day, start, end in intervals:
            if day == now.weekday() and start <= cur_min < end:
                return True, None

        # Find the next opening within a week.
        best: Optional[datetime] = None
        for ahead in range(0, 8):
            d = (now.weekday() + ahead) % 7
            for day, start, _end in sorted(intervals):
                if day != d:
                    continue
                candidate = (now + timedelta(days=ahead)).replace(
                    hour=start // 60, minute=start % 60, second=0, microsecond=0)
                if candidate > now and (best is None or candidate < best):
                    best = candidate
        return False, (best.timestamp() if best else None)


class DialAndBridgeChannel(PhoneChannel):
    """Non-autonomous fallback: don't call on the user's behalf — hand them the number."""
    is_async = False

    async def commit(self, plan: CommitPlan, payment: Any = None) -> BookingResult:
        phone = plan.details.get("phone") or "the business"
        return BookingResult(
            success=False, needs_manual=True,
            message=(f"I won't call on your behalf in this mode, sir. Please ring "
                     f"{plan.details.get('business_name')} at {phone} to book."),
            error="dial_and_bridge",
        )


def _parse_hours(hours) -> List[Tuple[int, int, int]]:
    """Normalize Yelp-style hours into (weekday, open_min, close_min). Best-effort."""
    out: List[Tuple[int, int, int]] = []
    if not hours:
        return out
    # Yelp Fusion: hours = [{"open": [{"day":0,"start":"1700","end":"2200"}, ...]}]
    blocks = hours if isinstance(hours, list) else [hours]
    for block in blocks:
        opens = block.get("open", []) if isinstance(block, dict) else []
        for o in opens:
            try:
                day = int(o["day"])
                start = int(o["start"][:2]) * 60 + int(o["start"][2:])
                end = int(o["end"][:2]) * 60 + int(o["end"][2:])
                if end <= start:  # overnight; clamp to end of day for simplicity
                    end = 24 * 60
                out.append((day, start, end))
            except (KeyError, ValueError, TypeError):
                continue
    return out
