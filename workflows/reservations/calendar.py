"""
CalendarService (M4).

On a confirmed booking, creates a Google Calendar event with the reservation
details. The event *insertion* is pluggable: the real Google inserter is wired
only when credentials are configured; otherwise it's a safe no-op.

Dates/times arrive already canonicalized by the harness (ISO `YYYY-MM-DD`,
24-hour `HH:MM` — core/harness/normalize.py), so this service does no parsing
of human phrasing; uncanonical input means "skip the event", never a guess.

Creating the event never fails the booking — failures are logged and ignored.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DURATION_MIN = 90


class CalendarService:
    def __init__(self, calendar_id: str = "primary",
                 inserter: Optional[Callable[[str, Dict[str, Any]], Optional[str]]] = None,
                 timezone: Optional[str] = None):
        self.calendar_id = calendar_id
        self._inserter = inserter           # callable(calendar_id, event_dict) -> link/id or None
        self.timezone = timezone

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

    # ------------------------------------------------------------------ canonical
    @staticmethod
    def _parse_when(date_str: Optional[str],
                    time_str: Optional[str]) -> Optional[Tuple[datetime, datetime]]:
        """Canonical 'YYYY-MM-DD' + 'HH:MM' → (start, end). Anything else → None."""
        if not date_str or not time_str:
            return None
        try:
            start = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            return None
        return start, start + timedelta(minutes=_DURATION_MIN)


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
