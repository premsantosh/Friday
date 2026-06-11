"""
CalendarService (M4).

On a confirmed booking, creates a Google Calendar event with the reservation
details. The event *insertion* is pluggable: the real Google inserter is wired
only when credentials are configured; otherwise it's a safe no-op. Building the
event dict and parsing the human date/time are deterministic and tested.

Creating the event never fails the booking — failures are logged and ignored.
"""

from __future__ import annotations

import calendar as _calendar
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DURATION_MIN = 90
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_TIME_RE = re.compile(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", re.I)


class CalendarService:
    def __init__(self, calendar_id: str = "primary",
                 inserter: Optional[Callable[[str, Dict[str, Any]], Optional[str]]] = None,
                 timezone: Optional[str] = None,
                 now: Optional[Callable[[], datetime]] = None):
        self.calendar_id = calendar_id
        self._inserter = inserter           # callable(calendar_id, event_dict) -> link/id or None
        self.timezone = timezone
        self._now = now or datetime.now     # injectable for deterministic tests

    @classmethod
    def from_env(cls) -> "CalendarService":
        cal_id = os.getenv("RESERVATION_CALENDAR_ID", "primary")
        tz = os.getenv("RESERVATION_TIMEZONE")
        return cls(cal_id, inserter=_google_inserter_from_env(), timezone=tz)

    # ------------------------------------------------------------------ public
    def create_event(self, facts: Dict[str, Any]) -> Optional[str]:
        """Build and insert the event. Returns a link/id, or None (no-op or failure)."""
        event = self.build_event(facts)
        if event is None:
            logger.info("Calendar: couldn't determine a date/time; skipping event.")
            return None
        if self._inserter is None:
            logger.info("Calendar not configured; skipping event creation.")
            return None
        try:
            return self._inserter(self.calendar_id, event)
        except Exception:
            logger.warning("Calendar event creation failed.", exc_info=True)
            return None

    def build_event(self, facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        when = self._parse_when(facts.get("date"), facts.get("time"))
        if when is None:
            return None
        start, end = when

        business = facts.get("business_name", "Reservation")
        party = facts.get("party_size")
        method = facts.get("method")
        confirmation = facts.get("confirmation")

        desc_lines = [f"Reservation at {business}."]
        if party:
            desc_lines.append(f"Party size: {party}.")
        if method:
            desc_lines.append(f"Booked via: {method}.")
        if confirmation:
            desc_lines.append(f"Confirmation: {confirmation}.")
        desc_lines.append("Booked by Friday.")

        event: Dict[str, Any] = {
            "summary": f"Reservation — {business}",
            "description": "\n".join(desc_lines),
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
        }
        if facts.get("address"):
            event["location"] = facts["address"]
        if self.timezone:
            event["start"]["timeZone"] = self.timezone
            event["end"]["timeZone"] = self.timezone
        return event

    # ------------------------------------------------------------------ parsing
    def _parse_when(self, date_str: Optional[str],
                    time_str: Optional[str]) -> Optional[Tuple[datetime, datetime]]:
        day = self._parse_date(date_str)
        tod = self._parse_time(time_str)
        if day is None or tod is None:
            return None
        start = day.replace(hour=tod[0], minute=tod[1], second=0, microsecond=0)
        return start, start + timedelta(minutes=_DURATION_MIN)

    def _parse_date(self, s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        s = s.strip().lower()
        today = self._now()
        if s in ("today", "tonight", "this evening", "this afternoon"):
            return today
        if s == "tomorrow":
            return today + timedelta(days=1)

        m = re.search(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", s)
        if m:
            month, dayn = int(m.group(1)), int(m.group(2))
            year = int(m.group(3)) if m.group(3) else today.year
            if year < 100:
                year += 2000
            try:
                return today.replace(year=year, month=month, day=dayn)
            except ValueError:
                return None

        for name, idx in _WEEKDAYS.items():
            if name in s:
                ahead = (idx - today.weekday()) % 7
                if ahead == 0 or "next" in s:
                    ahead = ahead or 7
                    if "next" in s and (idx - today.weekday()) % 7 != 0:
                        ahead = (idx - today.weekday()) % 7
                    else:
                        ahead = 7 if ahead == 0 else ahead
                return today + timedelta(days=ahead)
        return None

    @staticmethod
    def _parse_time(s: Optional[str]) -> Optional[Tuple[int, int]]:
        if not s:
            return None
        m = _TIME_RE.search(s.strip().lower())
        if not m:
            return None
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = m.group(3)
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        return hour, minute


def _google_inserter_from_env() -> Optional[Callable[[str, Dict[str, Any]], Optional[str]]]:
    """
    Build a real Google Calendar inserter if credentials are configured, else None.

    Friday has no Google auth wired in yet, so this is best-effort: it activates
    only when GOOGLE_APPLICATION_CREDENTIALS (a service-account JSON with calendar
    scope and access to RESERVATION_CALENDAR_ID) is present and the client library
    is installed. Otherwise the service no-ops.
    """
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        return None

    def _insert(calendar_id: str, event: Dict[str, Any]) -> Optional[str]:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            creds_path, scopes=["https://www.googleapis.com/auth/calendar.events"]
        )
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        created = service.events().insert(calendarId=calendar_id, body=event).execute()
        return created.get("htmlLink") or created.get("id")

    return _insert
