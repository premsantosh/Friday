"""
Time workflow - current time and timezone lookups.
Uses Python's built-in zoneinfo (no external dependencies).
"""

import re
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Any, Dict, Optional

from .base import Workflow, WorkflowResult, WorkflowStatus, WorkflowTrigger


# Mapping of common city/country/region names → IANA timezone IDs
TIMEZONE_MAP: Dict[str, str] = {
    # Cities - Americas
    "new york": "America/New_York",
    "los angeles": "America/Los_Angeles",
    "la": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "houston": "America/Chicago",
    "denver": "America/Denver",
    "phoenix": "America/Phoenix",
    "san francisco": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "miami": "America/New_York",
    "boston": "America/New_York",
    "toronto": "America/Toronto",
    "montreal": "America/Toronto",
    "vancouver": "America/Vancouver",
    "mexico city": "America/Mexico_City",
    "sao paulo": "America/Sao_Paulo",
    "buenos aires": "America/Argentina/Buenos_Aires",
    "bogota": "America/Bogota",
    "lima": "America/Lima",
    "santiago": "America/Santiago",
    "caracas": "America/Caracas",

    # Cities - Europe
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "madrid": "Europe/Madrid",
    "rome": "Europe/Rome",
    "amsterdam": "Europe/Amsterdam",
    "brussels": "Europe/Brussels",
    "zurich": "Europe/Zurich",
    "vienna": "Europe/Vienna",
    "stockholm": "Europe/Stockholm",
    "oslo": "Europe/Oslo",
    "copenhagen": "Europe/Copenhagen",
    "helsinki": "Europe/Helsinki",
    "warsaw": "Europe/Warsaw",
    "prague": "Europe/Prague",
    "budapest": "Europe/Budapest",
    "bucharest": "Europe/Bucharest",
    "athens": "Europe/Athens",
    "istanbul": "Europe/Istanbul",
    "kyiv": "Europe/Kyiv",
    "moscow": "Europe/Moscow",
    "lisbon": "Europe/Lisbon",

    # Cities - Asia
    "dubai": "Asia/Dubai",
    "abu dhabi": "Asia/Dubai",
    "riyadh": "Asia/Riyadh",
    "tehran": "Asia/Tehran",
    "karachi": "Asia/Karachi",
    "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "new delhi": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata",
    "chennai": "Asia/Kolkata",
    "dhaka": "Asia/Dhaka",
    "colombo": "Asia/Colombo",
    "kathmandu": "Asia/Kathmandu",
    "islamabad": "Asia/Karachi",
    "kabul": "Asia/Kabul",
    "tashkent": "Asia/Tashkent",
    "almaty": "Asia/Almaty",
    "bangkok": "Asia/Bangkok",
    "jakarta": "Asia/Jakarta",
    "singapore": "Asia/Singapore",
    "kuala lumpur": "Asia/Kuala_Lumpur",
    "manila": "Asia/Manila",
    "hong kong": "Asia/Hong_Kong",
    "taipei": "Asia/Taipei",
    "shanghai": "Asia/Shanghai",
    "beijing": "Asia/Shanghai",
    "chongqing": "Asia/Shanghai",
    "seoul": "Asia/Seoul",
    "tokyo": "Asia/Tokyo",
    "osaka": "Asia/Tokyo",
    "ho chi minh": "Asia/Ho_Chi_Minh",
    "hanoi": "Asia/Bangkok",
    "yangon": "Asia/Yangon",
    "phnom penh": "Asia/Phnom_Penh",
    "ulaanbaatar": "Asia/Ulaanbaatar",

    # Cities - Africa / Middle East
    "cairo": "Africa/Cairo",
    "nairobi": "Africa/Nairobi",
    "lagos": "Africa/Lagos",
    "johannesburg": "Africa/Johannesburg",
    "cape town": "Africa/Johannesburg",
    "casablanca": "Africa/Casablanca",
    "accra": "Africa/Accra",
    "addis ababa": "Africa/Addis_Ababa",
    "khartoum": "Africa/Khartoum",
    "tel aviv": "Asia/Jerusalem",
    "jerusalem": "Asia/Jerusalem",
    "amman": "Asia/Amman",
    "beirut": "Asia/Beirut",
    "baghdad": "Asia/Baghdad",
    "muscat": "Asia/Muscat",
    "doha": "Asia/Qatar",
    "kuwait": "Asia/Kuwait",

    # Cities - Oceania
    "sydney": "Australia/Sydney",
    "melbourne": "Australia/Melbourne",
    "brisbane": "Australia/Brisbane",
    "perth": "Australia/Perth",
    "adelaide": "Australia/Adelaide",
    "auckland": "Pacific/Auckland",
    "wellington": "Pacific/Auckland",
    "honolulu": "Pacific/Honolulu",
    "fiji": "Pacific/Fiji",

    # Countries
    "usa": "America/New_York",
    "us": "America/New_York",
    "united states": "America/New_York",
    "america": "America/New_York",
    "canada": "America/Toronto",
    "mexico": "America/Mexico_City",
    "brazil": "America/Sao_Paulo",
    "argentina": "America/Argentina/Buenos_Aires",
    "chile": "America/Santiago",
    "colombia": "America/Bogota",
    "peru": "America/Lima",
    "uk": "Europe/London",
    "united kingdom": "Europe/London",
    "england": "Europe/London",
    "britain": "Europe/London",
    "scotland": "Europe/London",
    "ireland": "Europe/London",
    "france": "Europe/Paris",
    "germany": "Europe/Berlin",
    "spain": "Europe/Madrid",
    "italy": "Europe/Rome",
    "netherlands": "Europe/Amsterdam",
    "switzerland": "Europe/Zurich",
    "sweden": "Europe/Stockholm",
    "norway": "Europe/Oslo",
    "denmark": "Europe/Copenhagen",
    "finland": "Europe/Helsinki",
    "poland": "Europe/Warsaw",
    "russia": "Europe/Moscow",
    "ukraine": "Europe/Kyiv",
    "turkey": "Europe/Istanbul",
    "greece": "Europe/Athens",
    "portugal": "Europe/Lisbon",
    "uae": "Asia/Dubai",
    "saudi arabia": "Asia/Riyadh",
    "india": "Asia/Kolkata",
    "pakistan": "Asia/Karachi",
    "bangladesh": "Asia/Dhaka",
    "china": "Asia/Shanghai",
    "japan": "Asia/Tokyo",
    "south korea": "Asia/Seoul",
    "korea": "Asia/Seoul",
    "thailand": "Asia/Bangkok",
    "vietnam": "Asia/Ho_Chi_Minh",
    "indonesia": "Asia/Jakarta",
    "malaysia": "Asia/Kuala_Lumpur",
    "philippines": "Asia/Manila",
    "australia": "Australia/Sydney",
    "new zealand": "Pacific/Auckland",
    "egypt": "Africa/Cairo",
    "kenya": "Africa/Nairobi",
    "nigeria": "Africa/Lagos",
    "south africa": "Africa/Johannesburg",
    "israel": "Asia/Jerusalem",

    # Common abbreviations / offsets
    "utc": "UTC",
    "gmt": "UTC",
    "est": "America/New_York",
    "cst": "America/Chicago",
    "mst": "America/Denver",
    "pst": "America/Los_Angeles",
    "ist": "Asia/Kolkata",
    "jst": "Asia/Tokyo",
    "cet": "Europe/Paris",
    "aest": "Australia/Sydney",
}


def _find_timezone(location: str) -> Optional[ZoneInfo]:
    """Resolve a location string to a ZoneInfo object."""
    loc = location.strip().lower()

    # Direct lookup in our map
    if loc in TIMEZONE_MAP:
        return ZoneInfo(TIMEZONE_MAP[loc])

    # Try it as a raw IANA timezone ID (e.g. "America/Chicago")
    try:
        return ZoneInfo(location.strip())
    except ZoneInfoNotFoundError:
        pass

    # Partial match on whole words only — substring matching mapped "Atlanta"
    # to "la" (Los Angeles) and "Austin" to "us". A key matches when the query
    # words contain all of the key's words (or vice versa for multi-word keys).
    loc_words = set(loc.split())
    for key, tz_id in TIMEZONE_MAP.items():
        key_words = key.split()
        if set(key_words) <= loc_words or loc_words <= set(key_words):
            return ZoneInfo(tz_id)

    return None


def _format_time(dt: datetime) -> str:
    """Return a natural-sounding time string, e.g. '3:07 PM'."""
    return dt.strftime("%-I:%M %p")


def _format_datetime(dt: datetime) -> str:
    """Return date + time, e.g. 'Wednesday, 25 February at 3:07 PM'."""
    return dt.strftime("%A, %-d %B at %-I:%M %p")


class TimeWorkflow(Workflow):
    """Handles time queries — current local time and time in any location."""

    @property
    def name(self) -> str:
        return "time"

    @property
    def description(self) -> str:
        return "Tell the current time, or the time in any city, country, or timezone."

    @property
    def trigger(self) -> WorkflowTrigger:
        return WorkflowTrigger(
            keywords=["time", "clock", "hour", "timezone", "o'clock"],
            patterns=[
                r"what(?:'s| is) the time",
                r"what time is it",
                r"current time",
                r"time in .+",
                r"time (at|for) .+",
                r"what(?:'s| is).+time.+in .+",
            ],
            examples=[
                "What time is it?",
                "What's the current time?",
                "What time is it in Tokyo?",
                "What's the time in London?",
                "What time is it in India?",
                "Time in New York",
            ],
        )

    async def execute(self, intent: str, entities: Dict[str, Any]) -> WorkflowResult:
        text = intent.lower()

        # Detect a location in the query
        location = self._extract_location(text)

        if location:
            tz = _find_timezone(location)
            if tz is None:
                return WorkflowResult(
                    status=WorkflowStatus.FAILURE,
                    message=(
                        f"I'm afraid I don't have timezone data for '{location}', sir. "
                        "Perhaps try a major city nearby?"
                    ),
                )
            now = datetime.now(tz)
            tz_label = str(tz)
            friendly = location.title()
            msg = (
                f"The current time in {friendly} is {_format_time(now)}, sir. "
                f"That's {_format_datetime(now)} ({tz_label})."
            )
        else:
            # Local time
            now = datetime.now().astimezone()
            msg = (
                f"The current time is {_format_time(now)}, sir. "
                f"Today is {_format_datetime(now)}."
            )

        return WorkflowResult(
            status=WorkflowStatus.SUCCESS,
            message=msg,
            data={"time": now.isoformat()},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _LOCATION_PATTERNS = [
        r"what.+time.+in (.+?)(?:\?|$)",   # "what time is it in Tokyo"
        r"time in (.+?)(?:\?|$)",            # "time in London"
        r"time at (.+?)(?:\?|$)",            # "time at Tokyo"
        r"time for (.+?)(?:\?|$)",           # "time for Japan"
        r"what(?:'s| is) (?:the )?(.+?) time",  # "what's the Tokyo time"
        r"\bin (.+?)(?:\?|$)",              # fallback: "in <location>"
    ]

    def _extract_location(self, text: str) -> Optional[str]:
        for pattern in self._LOCATION_PATTERNS:
            m = re.search(pattern, text)
            if m:
                candidate = m.group(1).strip()
                # Filter out filler words that aren't locations
                if candidate not in ("it", "the", "now", "current", "local", ""):
                    return candidate
        return None
