"""
TableCheckChannel (M8) — availability polling + booking hand-off for venues on
tablecheck.com (e.g. https://www.tablecheck.com/en/benfiddich-tokyo/reserve).

Unlike the browser channels this one is pure HTTP: the TableCheck booking
widget is backed by a JSON availability API, so `check_availability` returns a
*concrete* AVAILABLE/UNAVAILABLE — the first channel that does, which is what
makes the watcher able to actually fire. `commit` is a safe hand-off with a
deep link (no auto-booking in v1).

ENDPOINT CONTRACT — verify against live widget traffic (plan M0):
The default URL template and the response shapes accepted by `_parse_raw`
encode the documented web_booking API pattern. Before relying on this in
production, capture the real request the reserve widget makes (browser
devtools → XHR on the /reserve page), then:
  * set TABLECHECK_AVAILABILITY_URL if the path/params differ
    (placeholders: {slug}, {party_size}, {start_date}, {end_date}), and
  * refresh tests/fixtures/tablecheck/*.json with real responses —
    `_parse_raw` is deliberately strict and raises SchemaDrift on anything it
    doesn't recognise, which pauses the watch instead of silently reporting
    "no availability".
"""

from __future__ import annotations

import logging
import os
import re
import urllib.parse
from calendar import monthrange
from typing import Any, Callable, Dict, Optional, Tuple

from core.harness import normalize_time

from ..models import ChannelDecision, ReservationMethod
from .base import (
    Availability,
    AvailabilityStatus,
    BookingResult,
    CommitPlan,
    ReservationChannel,
)

logger = logging.getLogger(__name__)

DEFAULT_AVAILABILITY_URL = (
    "https://api.tablecheck.com/api/web_booking/v1/shops/{slug}/availability_calendar"
    "?num_people={party_size}&start_date={start_date}&end_date={end_date}"
)

# Browser-equivalent headers: this is the same traffic the widget sends, at a
# far lower rate than a human clicking through months (ToS etiquette, plan §risks).
_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en",
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
}

_HALTED_VALUES = frozenset({"halted", "paused", "suspended", "closed_temporarily"})

_SLUG_RE = re.compile(
    r"tablecheck\.com/(?:[a-z]{2}(?:-[a-z]{2})?/)?(?:shops?/)?([a-z0-9][a-z0-9\-_]+)",
    re.I,
)


class TableCheckError(Exception):
    """HTTP-level failure (network error, 429, 5xx…). `status` is the HTTP
    status code when there was a response, else None."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class SchemaDrift(TableCheckError):
    """The response parsed as JSON but didn't match the known shape — the
    endpoint has likely changed. The watcher pauses + alerts on this rather
    than treating it as 'no availability' (drift is the #1 failure mode)."""


def slug_from_url(url: Optional[str]) -> Optional[str]:
    """'https://www.tablecheck.com/en/benfiddich-tokyo/reserve' → 'benfiddich-tokyo'."""
    if not url:
        return None
    m = _SLUG_RE.search(url)
    return m.group(1) if m else None


class TableCheckChannel(ReservationChannel):
    method = ReservationMethod.TABLECHECK
    can_commit = False   # v1 hands off with a deep link; no auto-booking

    def __init__(self, fetcher: Optional[Callable] = None):
        # fetcher: async callable(url, headers) -> (status_code, json_or_None).
        # Injectable for tests; the default uses aiohttp (guarded import).
        self._fetcher = fetcher
        self.url_template = os.getenv("TABLECHECK_AVAILABILITY_URL",
                                      DEFAULT_AVAILABILITY_URL)

    # ------------------------------------------------------------------ fetch
    async def fetch_month(self, slug: str, party_size: int, month: str) -> Dict[str, Any]:
        """Normalized availability snapshot for one (venue, party size, month)
        fetch unit — the primitive the watcher polls.

        Raises TableCheckError on HTTP/network trouble, SchemaDrift when the
        response shape isn't recognised.
        """
        year, mon = (int(x) for x in month.split("-"))
        last_day = monthrange(year, mon)[1]
        url = self.url_template.format(
            slug=urllib.parse.quote(slug), party_size=int(party_size),
            start_date=f"{month}-01", end_date=f"{month}-{last_day:02d}")
        status, body = await self._fetch(url)
        if status == 429:
            raise TableCheckError("Rate limited by TableCheck (429).", status=429)
        if status >= 500:
            raise TableCheckError(f"TableCheck server error ({status}).", status=status)
        if status != 200:
            # Unexpected 4xx is drift territory: the endpoint moved or the slug
            # scheme changed — don't keep hammering it.
            raise SchemaDrift(f"Unexpected HTTP {status} from availability endpoint.",
                              status=status)
        return self._parse_raw(body, month)

    async def _fetch(self, url: str) -> Tuple[int, Any]:
        if self._fetcher is not None:
            return await self._fetcher(url, _HEADERS)
        try:
            import aiohttp
        except ImportError as exc:
            raise TableCheckError(f"aiohttp is not installed: {exc}") from exc
        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.get(url, headers=_HEADERS) as resp:
                    if resp.status != 200:
                        return resp.status, None
                    return resp.status, await resp.json(content_type=None)
        except TableCheckError:
            raise
        except Exception as exc:
            raise TableCheckError(f"Availability request failed: {exc}") from exc

    # -------------------------------------------------------------- normalize
    @staticmethod
    def _parse_raw(raw: Any, month: str) -> Dict[str, Any]:
        """Raw response → the normalized snapshot shape the differ consumes:
        {"month", "dates": {iso: {"open", "slots": {"HH:MM": "available"}}},
         "venue_status": "ok"|"halted"}.
        Only real data — a date/slot the response doesn't mention is absent,
        never inferred. Deterministic ordering via sorted keys downstream.
        """
        if not isinstance(raw, dict):
            raise SchemaDrift("Availability response is not a JSON object.")

        venue_status = "ok"
        shop = raw.get("shop") if isinstance(raw.get("shop"), dict) else {}
        for candidate in (shop.get("status"), raw.get("venue_status"), raw.get("status")):
            if isinstance(candidate, str) and candidate.lower() in _HALTED_VALUES:
                venue_status = "halted"

        calendar = None
        for key in ("availability_calendar", "availability", "calendar"):
            if isinstance(raw.get(key), dict):
                calendar = raw[key]
                break
        days_list = raw.get("available_days")

        dates: Dict[str, Dict[str, Any]] = {}
        if calendar is not None:
            for day, value in calendar.items():
                parsed = TableCheckChannel._parse_day(day, value)
                if parsed is not None:
                    dates[day] = parsed
        elif isinstance(days_list, list):
            for entry in days_list:
                if not isinstance(entry, dict) or "date" not in entry:
                    raise SchemaDrift("available_days entry without a date.")
                value = entry.get("times", entry.get("slots", entry.get("available")))
                parsed = TableCheckChannel._parse_day(entry["date"], value)
                if parsed is not None:
                    dates[entry["date"]] = parsed
        elif venue_status != "halted":
            raise SchemaDrift("No recognised availability structure in response.")

        return {"month": month, "dates": dates, "venue_status": venue_status}

    @staticmethod
    def _parse_day(day: Any, value: Any) -> Optional[Dict[str, Any]]:
        """One calendar entry → {"open", "slots"}; None for a malformed date key."""
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(day)):
            return None
        slots: Dict[str, str] = {}
        if isinstance(value, list):
            for t in value:
                norm = normalize_time(str(t))
                if norm:
                    slots[norm] = "available"
            return {"open": bool(slots), "slots": slots}
        if isinstance(value, dict):
            for t, state in value.items():
                norm = normalize_time(str(t))
                if not norm:
                    continue
                is_open = state in (True, "available", "open")
                slots[norm] = "available" if is_open else "full"
            return {"open": any(s == "available" for s in slots.values()),
                    "slots": slots}
        if isinstance(value, bool):
            return {"open": value, "slots": {}}   # day-level source
        raise SchemaDrift(f"Unrecognised calendar value for {day!r}.")

    # ------------------------------------------------------------- deep links
    @staticmethod
    def booking_url(slug: str, date: Optional[str] = None,
                    time: Optional[str] = None,
                    party_size: Optional[int] = None) -> str:
        """Reserve-page deep link. Query params follow the widget's URL scheme —
        verify alongside the endpoint (M0); the bare /reserve page works even
        if the params are ignored."""
        params = {}
        if date:
            params["start_date"] = date
        if time:
            params["start_time"] = time
        if party_size:
            params["num_people"] = str(party_size)
        query = ("?" + urllib.parse.urlencode(params)) if params else ""
        return f"https://www.tablecheck.com/en/{urllib.parse.quote(slug)}/reserve{query}"

    # ----------------------------------------------------------- channel ABC
    async def check_availability(self, slots: Dict[str, Any]) -> Availability:
        """Concrete availability for the booking flow's (date, time, party).
        UNKNOWN only when we genuinely can't tell (no slug, fetch trouble,
        venue halted)."""
        slug = self._slug_from_slots(slots)
        date, time_hhmm = slots.get("date"), slots.get("time")
        party = slots.get("party_size")
        if not (slug and date and party):
            return Availability(status=AvailabilityStatus.UNKNOWN,
                                note="I couldn't identify the TableCheck venue page.")
        try:
            state = await self.fetch_month(slug, int(party), str(date)[:7])
        except TableCheckError as exc:
            logger.warning("TableCheck availability check failed: %s", exc)
            return Availability(status=AvailabilityStatus.UNKNOWN,
                                note="TableCheck wasn't reachable just now.")
        if state["venue_status"] == "halted":
            return Availability(
                status=AvailabilityStatus.UNKNOWN,
                note="They've temporarily halted accepting new reservations.")

        info = state["dates"].get(str(date)) or {}
        available = sorted(t for t, s in (info.get("slots") or {}).items()
                           if s == "available")
        if info.get("open") and (not time_hhmm or time_hhmm in available
                                 or not info.get("slots")):
            return Availability(status=AvailabilityStatus.AVAILABLE, options=available)
        return Availability(status=AvailabilityStatus.UNAVAILABLE, options=available)

    async def prepare(self, slots: Dict[str, Any], decision: ChannelDecision) -> CommitPlan:
        slug = self._slug_from_slots(slots) or slug_from_url(decision.url) or ""
        url = self.booking_url(slug, slots.get("date"), slots.get("time"),
                               slots.get("party_size")) if slug else decision.url
        summary = (f"Book {decision.business_name or slots.get('business_name')} via "
                   f"TableCheck for {slots.get('party_size', 'your party')} on "
                   f"{slots.get('date')} at {slots.get('time')}")
        return CommitPlan(
            channel="tablecheck", summary=summary,
            details={"url": url, "slug": slug,
                     "business_name": decision.business_name or slots.get("business_name"),
                     "date": slots.get("date"), "time": slots.get("time"),
                     "party_size": slots.get("party_size")},
            requires_card=decision.requires_card_hint,
        )

    async def commit(self, plan: CommitPlan, payment: Any = None) -> BookingResult:
        # v1 never books TableCheck automatically — the whole point of the
        # watcher is a human on a fast notification with a ready deep link.
        return BookingResult(
            success=False, needs_manual=True,
            message=(f"TableCheck bookings need your hand, sir — the slot is held "
                     f"here: {plan.details.get('url')}"),
            error="tablecheck_manual_only",
        )

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _slug_from_slots(slots: Dict[str, Any]) -> Optional[str]:
        decision = slots.get("channel_decision") or {}
        return (slug_from_url(decision.get("url"))
                or slug_from_url(slots.get("target_url")))
