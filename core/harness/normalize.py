"""
Slot normalization — canonicalize fuzzy values at the boundary
(harness spec §7).

Relative dates ("next Friday") are resolved against *now* once, at slot-fill
time, so what the user approves — and what every downstream consumer
(calendar, email, call plan, watch deadline) reads — is a concrete canonical
value: dates as ISO `YYYY-MM-DD`, times as 24-hour `HH:MM`.

Every normalizer is a pure function returning the canonical value or None;
None means "re-ask the user", never "store the raw string and hope".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Optional

MAX_DAYS_AHEAD = 366

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

_TIME_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?\b", re.I)
_HALF_PAST_RE = re.compile(r"\bhalf\s+past\s+(\w+)\b", re.I)
_QUARTER_RE = re.compile(r"\bquarter\s+(past|to)\s+(\w+)\b", re.I)
_OCLOCK_RE = re.compile(r"\b(\w+)\s+o'?clock\b", re.I)
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")


@dataclass(frozen=True)
class NormalizeCtx:
    """Injected clock, so resolution is deterministic and testable."""
    now: datetime

    @classmethod
    def fresh(cls) -> "NormalizeCtx":
        return cls(now=datetime.now())


@dataclass(frozen=True)
class SlotSpec:
    """A collectible slot: how to ask for it, how to re-ask after a value the
    normalizer rejected, and the normalizer that canonicalizes it."""
    name: str
    prompt: str
    reask: str
    normalize: Callable[[str, NormalizeCtx], Optional[Any]]
    required: bool = True


# -------------------------------------------------------------------- string

def normalize_text(text: str, ctx: Optional[NormalizeCtx] = None) -> Optional[str]:
    cleaned = text.strip(" .,!?\"'")
    return cleaned if cleaned else None


# ---------------------------------------------------------------------- date

def normalize_date(text: str, ctx: NormalizeCtx) -> Optional[str]:
    """Resolve to ISO YYYY-MM-DD. None when unparseable, in the past, or more
    than a year out."""
    if not text or not text.strip():
        return None
    s = text.strip().lower()
    today = ctx.now.date()

    resolved = (_relative_date(s, today)
                or _weekday_date(s, today)
                or _slash_date(s, today)
                or _dateutil_date(s, ctx.now))
    if resolved is None:
        return None
    if resolved < today or (resolved - today).days > MAX_DAYS_AHEAD:
        return None
    return resolved.isoformat()


def _relative_date(s: str, today: date) -> Optional[date]:
    if s in ("today", "tonight", "this evening", "this afternoon"):
        return today
    if s == "tomorrow":
        return today + timedelta(days=1)
    if s in ("this weekend", "the weekend"):
        ahead = (5 - today.weekday()) % 7   # next Saturday (today if Saturday)
        return today + timedelta(days=ahead)
    return None


def _weekday_date(s: str, today: date) -> Optional[date]:
    for idx, name in enumerate(_WEEKDAYS):
        if re.search(rf"\b{name}\b", s):
            ahead = (idx - today.weekday()) % 7
            if "next" in s:
                # "next <weekday>": the one in the coming week, never today.
                ahead = ahead or 7
            elif ahead == 0 and "this" not in s:
                ahead = 7   # bare weekday on that same weekday → next week's
            return today + timedelta(days=ahead)
    return None


def _slash_date(s: str, today: date) -> Optional[date]:
    m = _SLASH_DATE_RE.search(s)
    if not m:
        return None
    month, day = int(m.group(1)), int(m.group(2))
    year = int(m.group(3)) if m.group(3) else today.year
    if year < 100:
        year += 2000
    try:
        candidate = date(year, month, day)
    except ValueError:
        return None
    # M/D with no explicit year that already passed → they mean next year.
    if m.group(3) is None and candidate < today:
        try:
            candidate = date(year + 1, month, day)
        except ValueError:
            return None
    return candidate


def _dateutil_date(s: str, now: datetime) -> Optional[date]:
    try:
        from dateutil import parser as date_parser
    except ImportError:
        return None
    try:
        parsed = date_parser.parse(s, default=now, fuzzy=False, dayfirst=False)
    except (ValueError, OverflowError):
        return None
    candidate = parsed.date()
    # "June 1" with no year that already passed → next year.
    if candidate < now.date() and not re.search(r"\b\d{4}\b", s):
        candidate = date(candidate.year + 1, candidate.month, candidate.day)
    return candidate


# ---------------------------------------------------------------------- time

def normalize_time(text: str, ctx: Optional[NormalizeCtx] = None) -> Optional[str]:
    """Resolve to 24-hour HH:MM. Bare 1–11 with no am/pm is read as evening —
    the deterministic reservation-hours rule, shown back to the user at the
    confirmation gate before anything acts on it."""
    if not text or not text.strip():
        return None
    s = text.strip().lower()

    if "noon" in s or "midday" in s:
        return "12:00"
    if "midnight" in s:
        return "00:00"

    m = _HALF_PAST_RE.search(s)
    if m:
        hour = _word_hour(m.group(1))
        return _fmt(hour, 30, _meridiem(s)) if hour is not None else None
    m = _QUARTER_RE.search(s)
    if m:
        hour = _word_hour(m.group(2))
        if hour is None:
            return None
        if m.group(1) == "to":
            return _fmt(hour - 1 if hour > 1 else 12, 45, _meridiem(s))
        return _fmt(hour, 15, _meridiem(s))
    m = _OCLOCK_RE.search(s)
    if m:
        hour = _word_hour(m.group(1))
        return _fmt(hour, 0, _meridiem(s)) if hour is not None else None

    m = _TIME_RE.search(s)
    if m and (m.group(2) or m.group(3) or m.group(1)):
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        return _fmt(hour, minute, m.group(3) or _meridiem(s))
    return None


def _meridiem(s: str) -> Optional[str]:
    if re.search(r"\b(morning|a\.?m\.?)\b", s):
        return "am"
    if re.search(r"\b(evening|afternoon|night|p\.?m\.?)\b", s):
        return "pm"
    return None


def _word_hour(word: str) -> Optional[int]:
    if word.isdigit():
        h = int(word)
        return h if 1 <= h <= 12 else None
    return _NUMBER_WORDS.get(word.lower())


def _fmt(hour: int, minute: int, ampm: Optional[str]) -> Optional[str]:
    if ampm:
        ampm = ampm.replace(".", "").lower()
        if ampm.startswith("p") and hour < 12:
            hour += 12
        elif ampm.startswith("a") and hour == 12:
            hour = 0
    elif 1 <= hour <= 11:
        hour += 12   # evening assumption for bare reservation hours
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return f"{hour:02d}:{minute:02d}"


# --------------------------------------------------------------- party / etc.

def normalize_party_size(text: str, ctx: Optional[NormalizeCtx] = None) -> Optional[int]:
    s = text.strip().lower()
    m = re.search(r"\d{1,3}", s)
    if m:
        n = int(m.group(0))
        return n if 1 <= n <= 100 else None
    for word, n in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", s):
            return n
    return None


def normalize_email(text: str, ctx: Optional[NormalizeCtx] = None) -> Optional[str]:
    cleaned = text.strip().strip(".,;")
    return cleaned if _EMAIL_RE.match(cleaned) else None


def normalize_phone(text: str, ctx: Optional[NormalizeCtx] = None) -> Optional[str]:
    digits = re.sub(r"[^\d+]", "", text.strip())
    bare = digits.lstrip("+")
    if not (10 <= len(bare) <= 15 and bare.isdigit()):
        return None
    return ("+" + bare) if digits.startswith("+") else bare


# ------------------------------------------------------------------- display

def display_date(iso: Any) -> str:
    """'2026-06-19' -> 'Friday, June 19'. Non-ISO input is returned as-is."""
    try:
        d = date.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return str(iso)
    return f"{d:%A, %B} {d.day}"


def display_time(hhmm: Any) -> str:
    """'19:00' -> '7:00 pm'. Non-canonical input is returned as-is."""
    try:
        t = datetime.strptime(str(hhmm), "%H:%M")
    except (TypeError, ValueError):
        return str(hhmm)
    return f"{t:%-I:%M %p}".lower()
