"""
Reservation sniping (Phase 1) — schedule a best-effort booking attempt for the
instant a venue's reservation window opens.

Pure, side-effect-free helpers for the time maths so they're trivially testable.
The workflow owns the scheduling (a WAITING session) and the race; this module
only answers "given a dining date and a release policy, when (UTC) does the
window open, and how do I describe that to the user?".
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

# Common spoken/written abbreviations → IANA zones. US-centric by design; an
# explicit IANA name (e.g. "Europe/London") is always accepted as-is.
_TZ_ALIASES = {
    "ET": "America/New_York", "ET/EDT": "America/New_York",
    "EST": "America/New_York", "EDT": "America/New_York", "EASTERN": "America/New_York",
    "CT": "America/Chicago", "CST": "America/Chicago", "CDT": "America/Chicago",
    "CENTRAL": "America/Chicago",
    "MT": "America/Denver", "MST": "America/Denver", "MDT": "America/Denver",
    "MOUNTAIN": "America/Denver",
    "PT": "America/Los_Angeles", "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles", "PACIFIC": "America/Los_Angeles",
    "UTC": "UTC", "GMT": "UTC",
}


def resolve_timezone(name: Optional[str], default: Optional[str] = None) -> ZoneInfo:
    """First usable zone among: explicit name, caller default (e.g. the venue's
    own tz), the configured RESERVATION_TIMEZONE, then Pacific. Aliases like 'ET'
    and phrasings like 'Eastern Time' are mapped to IANA."""
    for candidate in (name, default, os.getenv("RESERVATION_TIMEZONE"), "America/Los_Angeles"):
        if not candidate:
            continue
        key = candidate.strip()
        # Normalise "Eastern (Standard|Daylight) Time" → "EASTERN" before lookup.
        norm = re.sub(r"\s+(standard|daylight)?\s*time$", "", key, flags=re.I).strip().upper()
        iana = _TZ_ALIASES.get(norm) or _TZ_ALIASES.get(key.upper(), key)
        try:
            return ZoneInfo(iana)
        except (ZoneInfoNotFoundError, ValueError, OSError):
            logger.debug("Unrecognised timezone %r; trying next fallback", candidate)
    return ZoneInfo("UTC")


# Coarse US location → timezone, so an out-of-town venue's window isn't computed
# in the user's own zone when the release tz is unstated. Approximate by design
# (a few states straddle zones); an explicit release tz always takes precedence.
_STATE_TZ = {
    "ny": "America/New_York", "nj": "America/New_York", "ma": "America/New_York",
    "pa": "America/New_York", "dc": "America/New_York", "fl": "America/New_York",
    "ga": "America/New_York", "ct": "America/New_York", "va": "America/New_York",
    "nc": "America/New_York", "sc": "America/New_York", "mi": "America/New_York",
    "oh": "America/New_York", "me": "America/New_York", "md": "America/New_York",
    "il": "America/Chicago", "tx": "America/Chicago", "mn": "America/Chicago",
    "mo": "America/Chicago", "wi": "America/Chicago", "la": "America/Chicago",
    "tn": "America/Chicago", "ok": "America/Chicago", "ks": "America/Chicago",
    "co": "America/Denver", "ut": "America/Denver", "nm": "America/Denver",
    "az": "America/Phoenix",
    "ca": "America/Los_Angeles", "wa": "America/Los_Angeles",
    "or": "America/Los_Angeles", "nv": "America/Los_Angeles",
}
_CITY_TZ = {
    "new york": "America/New_York", "nyc": "America/New_York", "brooklyn": "America/New_York",
    "boston": "America/New_York", "washington": "America/New_York", "miami": "America/New_York",
    "atlanta": "America/New_York", "philadelphia": "America/New_York",
    "chicago": "America/Chicago", "austin": "America/Chicago", "dallas": "America/Chicago",
    "houston": "America/Chicago", "new orleans": "America/Chicago", "nashville": "America/Chicago",
    "denver": "America/Denver", "phoenix": "America/Phoenix",
    "san francisco": "America/Los_Angeles", "los angeles": "America/Los_Angeles",
    "seattle": "America/Los_Angeles", "portland": "America/Los_Angeles",
    "san diego": "America/Los_Angeles", "las vegas": "America/Los_Angeles",
}


def timezone_for_location(text: Optional[str]) -> Optional[str]:
    """Best-effort IANA zone from a city/address string, else None."""
    if not text:
        return None
    low = text.lower()
    for city, tz in _CITY_TZ.items():
        if city in low:
            return tz
    m = re.search(r",\s*([a-z]{2})\b", low)   # "..., NY ..." style
    if m and m.group(1) in _STATE_TZ:
        return _STATE_TZ[m.group(1)]
    return None


def compute_release_fire_ts(dining_date_iso: str, days_ahead: int,
                            release_hhmm: str, tz: ZoneInfo) -> Optional[float]:
    """The UTC epoch when a rolling window opens: `days_ahead` before the dining
    date, at `release_hhmm` (24h "HH:MM") local to `tz`. None on bad input."""
    try:
        d = date.fromisoformat(dining_date_iso)
        hour, minute = (int(x) for x in release_hhmm.split(":"))
        open_day = d - timedelta(days=int(days_ahead))
        fire_local = datetime(open_day.year, open_day.month, open_day.day,
                              hour, minute, tzinfo=tz)
        return fire_local.timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


def describe_fire(fire_ts: float, tz: ZoneInfo) -> str:
    """Human-friendly local rendering of the fire instant, e.g.
    'Sat, May 31 at 10:00 AM EDT'."""
    dt = datetime.fromtimestamp(fire_ts, tz)
    return dt.strftime("%a, %b %-d at %-I:%M %p %Z")
