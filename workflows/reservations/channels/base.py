"""
Reservation channel interface + shared browser base (M2).

`prepare()` builds the object the confirmation gate shows the user (no side
effects). `commit()` is the only step that books, and the workflow calls it
*only after* the user approves — and never when the kill switch is set.

Live browser automation (Playwright) is an optional dependency and is wrapped
so a missing install, a missing card (payment lands in M3), or the kill switch
all degrade to a safe, honest hand-off instead of a fake success.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..models import ChannelDecision, ReservationMethod

logger = logging.getLogger(__name__)


def kill_switch_on() -> bool:
    return bool(os.getenv("RESERVATION_KILL_SWITCH"))


def persistent_context_options(profile_dir: str) -> Dict[str, Any]:
    """Launch kwargs shared by the booking channels and the login bootstrap.

    These reduce *false-positive* bot flagging when driving a site with the
    user's own logged-in account — they don't actively evade detection (the spec
    forbids CAPTCHA-solving / stealth; §5.1, §8.3):
      - `--disable-blink-features=AutomationControlled` drops the
        `navigator.webdriver` tell that Playwright otherwise sets.
      - RESERVATION_BROWSER_CHANNEL (e.g. "chrome", "msedge") uses the user's real
        installed browser instead of bundled Chromium, which fingerprints as a
        normal consumer browser. Empty → bundled Chromium.
    """
    opts: Dict[str, Any] = {
        "user_data_dir": profile_dir,
        "headless": False,
        "args": ["--disable-blink-features=AutomationControlled"],
    }
    channel = os.getenv("RESERVATION_BROWSER_CHANNEL", "").strip()
    if channel:
        opts["channel"] = channel
    return opts


class AvailabilityStatus(Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"      # couldn't verify ahead of time; will attempt at booking


@dataclass
class Availability:
    status: AvailabilityStatus = AvailabilityStatus.UNKNOWN
    options: List[str] = field(default_factory=list)  # alternative times, if offered
    note: str = ""


@dataclass
class CommitPlan:
    """Serializable description of what commit() will do — shown for approval."""
    channel: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    requires_card: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "summary": self.summary,
            "details": self.details,
            "requires_card": self.requires_card,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CommitPlan":
        return cls(
            channel=d["channel"], summary=d["summary"],
            details=d.get("details", {}), requires_card=d.get("requires_card", False),
        )


@dataclass
class BookingResult:
    success: bool
    message: str
    confirmation: Optional[str] = None
    error: Optional[str] = None
    needs_manual: bool = False   # we safely bailed; the user should finish it


class ReservationChannel(ABC):
    method: ReservationMethod = ReservationMethod.UNKNOWN
    can_commit: bool = False

    @abstractmethod
    async def check_availability(self, slots: Dict[str, Any]) -> Availability: ...

    @abstractmethod
    async def prepare(self, slots: Dict[str, Any], decision: ChannelDecision) -> CommitPlan: ...

    @abstractmethod
    async def commit(self, plan: CommitPlan, payment: Any = None) -> BookingResult: ...


class BrowserChannel(ReservationChannel):
    """
    Shared Playwright base. OpenTable/Resy/Yelp/GenericWeb differ only in their
    profile directory, target URL, and (later) site-specific booking logic.

    M2 establishes the browser-session lifecycle, the confirmation contract, and
    a generic best-effort form-filler (`_do_booking`). Per-site flows that need a
    logged-in account and bespoke selectors are layered in by overriding
    `_do_booking`; until then they hand off safely.
    """

    can_commit = True

    def __init__(self, method: ReservationMethod, platform: str, profile_dir: str):
        self.method = method
        self.platform = platform
        self.profile_dir = os.path.join(profile_dir, platform)

    async def check_availability(self, slots: Dict[str, Any]) -> Availability:
        # M2 doesn't reliably parse arbitrary slot grids ahead of time; we proceed
        # to confirmation and attempt the booking live.
        return Availability(
            status=AvailabilityStatus.UNKNOWN,
            note="I'll confirm the exact slot at the moment of booking.",
        )

    async def prepare(self, slots: Dict[str, Any], decision: ChannelDecision) -> CommitPlan:
        party = slots.get("party_size", "your party")
        summary = (
            f"Book {decision.business_name or slots.get('business_name')} via {self.platform} "
            f"for {party} on {slots.get('date')} at {slots.get('time')}"
        )
        details = {
            "url": decision.url,
            "business_name": decision.business_name or slots.get("business_name"),
            "date": slots.get("date"),
            "time": slots.get("time"),
            "party_size": slots.get("party_size"),
            "guest_name": slots.get("guest_name"),
            "phone": slots.get("phone"),
            "email": slots.get("email"),
        }
        return CommitPlan(
            channel=self.platform, summary=summary, details=details,
            requires_card=decision.requires_card_hint,
        )

    async def commit(self, plan: CommitPlan, payment: Any = None) -> BookingResult:
        if kill_switch_on():
            return BookingResult(
                success=False, needs_manual=True,
                message="I'm in research-only mode (kill switch on), sir, so I haven't booked it.",
                error="kill_switch",
            )
        if plan.requires_card and payment is None:
            # Payment (single-use Privacy card) is wired in M3.
            return BookingResult(
                success=False, needs_manual=True,
                message=(
                    f"{plan.details.get('business_name')} needs a card to hold the booking, sir. "
                    "Card handling isn't enabled yet, so you'll need to finish this one."
                ),
                error="card_required",
            )

        try:
            from playwright.async_api import async_playwright  # noqa: F401
        except Exception:
            return BookingResult(
                success=False, needs_manual=True,
                message=(
                    "Browser automation isn't installed, sir "
                    "(pip install playwright && playwright install chromium)."
                ),
                error="playwright_missing",
            )

        try:
            return await self._do_booking(plan)
        except Exception as exc:
            logger.exception("Browser commit failed on %s", self.platform)
            return BookingResult(
                success=False, needs_manual=True,
                message=(
                    f"I opened {self.platform} but couldn't complete the booking automatically, sir. "
                    f"Here's the link to finish it: {plan.details.get('url') or 'the booking page'}."
                ),
                error=str(exc),
            )

    async def _do_booking(self, plan: CommitPlan) -> BookingResult:
        """Override per site. Default: open the page, then hand off (no fake success)."""
        return BookingResult(
            success=False, needs_manual=True,
            message=(
                f"Automated booking for {self.platform} isn't implemented yet, sir — "
                f"it needs your logged-in session. Finish here: {plan.details.get('url')}."
            ),
            error="not_implemented",
        )

    # ------------------------------------------------------------------ helpers
    async def _launch(self, p):
        """Launch a persistent Chromium context so the user's login/cookies are reused."""
        os.makedirs(self.profile_dir, exist_ok=True)
        os.chmod(self.profile_dir, 0o700)  # session cookies are account-takeover tokens (§8 L3)
        return await p.chromium.launch_persistent_context(
            **persistent_context_options(self.profile_dir))
