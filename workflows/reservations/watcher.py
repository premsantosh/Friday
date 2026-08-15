"""
Availability watching (M8) — pure logic for multi-criteria reservation watches.

A watch is one venue plus a set of independent *criteria* — each with its own
date range, acceptable times, and party size ("Sep 5–6 at 7 or 9pm for two, and
also Sep 10 at 9pm for four"). This module owns the criteria model, the poll
planning (which fetches a cycle actually needs), the snapshot differ, and the
event×criteria matcher. Everything here is side-effect-free — the same
philosophy as `snipe.py` — so the tricky parts are trivially testable. The
workflow owns the scheduling (a WAITING session), the fetching (the TableCheck
channel), and the notifications.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from core.harness import NormalizeCtx, normalize_date, normalize_time

# --------------------------------------------------------------------- criteria


@dataclass
class WatchCriterion:
    """One thing the user actually wants: a date range × times × party size.

    Dates are ISO and venue-local; times are 24h HH:MM venue-local, or None
    for "any slot". Criteria have their own lifecycle — fulfilled by the user
    ("got the table"), expired when date_end passes in the venue's timezone.
    """
    date_start: str
    date_end: str
    party_size: int
    times: Optional[List[str]] = None     # None = any slot counts
    label: str = ""
    priority: int = 100                   # lower = more important
    status: str = "active"                # active|fulfilled|expired

    def to_dict(self) -> Dict[str, Any]:
        return {"date_start": self.date_start, "date_end": self.date_end,
                "party_size": self.party_size, "times": self.times,
                "label": self.label, "priority": self.priority, "status": self.status}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WatchCriterion":
        return cls(date_start=d["date_start"], date_end=d["date_end"],
                   party_size=int(d["party_size"]), times=d.get("times"),
                   label=d.get("label", ""), priority=int(d.get("priority", 100)),
                   status=d.get("status", "active"))

    def matches(self, slot_date: str, slot_time: Optional[str]) -> bool:
        """Does an opened (date, time) satisfy this criterion? A day-level event
        (time None — the source only said the *date* opened) matches even when
        specific times are wanted: better to notify than miss a cancellation."""
        if self.status != "active":
            return False
        if not (self.date_start <= slot_date <= self.date_end):
            return False
        if self.times is None or slot_time is None:
            return True
        return slot_time in self.times

    def describe(self) -> str:
        from core.harness import display_date, display_time
        if self.date_start == self.date_end:
            when = display_date(self.date_start)
        else:
            when = f"{display_date(self.date_start)} – {display_date(self.date_end)}"
        at = (" at " + " or ".join(display_time(t) for t in self.times)
              if self.times else " (any time)")
        tag = f" ({self.label})" if self.label else ""
        return f"{when}{at} for {self.party_size}{tag}"


def expire_criteria(criteria: List[WatchCriterion], today_iso: str) -> bool:
    """Expire criteria whose whole date range has passed (venue-local today).
    Returns True when anything changed."""
    changed = False
    for c in criteria:
        if c.status == "active" and c.date_end < today_iso:
            c.status = "expired"
            changed = True
    return changed


# ---------------------------------------------------------------- poll planning

def months_spanned(date_start: str, date_end: str) -> List[str]:
    """The 'YYYY-MM' months a date range touches, in order."""
    start, end = date.fromisoformat(date_start), date.fromisoformat(date_end)
    months, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def fetch_units(criteria: List[WatchCriterion]) -> List[Tuple[int, str]]:
    """Distinct (party_size, month) pairs the active criteria need, sorted.

    TableCheck availability is party-size-dependent, so each distinct party
    size is its own request stream; criteria sharing a unit are matched from
    the same snapshot. This is the dedup that keeps polling volume down.
    """
    units = {(c.party_size, month)
             for c in criteria if c.status == "active"
             for month in months_spanned(c.date_start, c.date_end)}
    return sorted(units)


# ---------------------------------------------------------------------- differ
#
# Normalized snapshot shape (produced by the channel's fetch_month):
#   {"month": "2026-09",
#    "dates": {"2026-09-05": {"open": True, "slots": {"19:00": "available"}}},
#    "venue_status": "ok" | "halted"}
# `slots` may be empty when the source only reports day-level availability.

def open_times(state: Dict[str, Any], date_iso: str) -> List[str]:
    """The available HH:MM times a snapshot reports for a date ([] when the
    date is closed or the source is day-level only)."""
    info = (state.get("dates") or {}).get(date_iso) or {}
    return sorted(t for t, s in (info.get("slots") or {}).items() if s == "available")


def open_dates(state: Dict[str, Any]) -> List[str]:
    return sorted(d for d, info in (state.get("dates") or {}).items()
                  if info.get("open"))


def diff_states(prev: Optional[Dict[str, Any]],
                curr: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Events between two snapshots of one fetch unit. The first snapshot is
    the baseline — no events, by design (we report *changes*, not state).

    Event kinds: slot_opened / slot_closed (time None for day-level sources),
    calendar_published, venue_halted / venue_resumed.
    """
    if prev is None:
        return []
    events: List[Dict[str, Any]] = []

    prev_halted = prev.get("venue_status") == "halted"
    curr_halted = curr.get("venue_status") == "halted"
    if curr_halted and not prev_halted:
        return [{"type": "venue_halted"}]
    if prev_halted and not curr_halted:
        events.append({"type": "venue_resumed"})

    # Published = the month carries date entries at all. A full-but-published
    # calendar freeing one table is a slot_opened, not a "publish".
    if not (prev.get("dates")) and curr.get("dates"):
        events.append({"type": "calendar_published", "dates": open_dates(curr)})

    all_dates = set((prev.get("dates") or {})) | set((curr.get("dates") or {}))
    for d in sorted(all_dates):
        p_times, c_times = set(open_times(prev, d)), set(open_times(curr, d))
        p_info = (prev.get("dates") or {}).get(d) or {}
        c_info = (curr.get("dates") or {}).get(d) or {}
        if p_times or c_times:
            for t in sorted(c_times - p_times):
                events.append({"type": "slot_opened", "date": d, "time": t})
            for t in sorted(p_times - c_times):
                events.append({"type": "slot_closed", "date": d, "time": t})
        else:
            # Day-level only: the source reports open dates without times.
            if c_info.get("open") and not p_info.get("open"):
                events.append({"type": "slot_opened", "date": d, "time": None})
            elif p_info.get("open") and not c_info.get("open"):
                events.append({"type": "slot_closed", "date": d, "time": None})
    return events


# --------------------------------------------------------------------- matcher

def match_events(events: List[Dict[str, Any]],
                 criteria: List[WatchCriterion]) -> List[Tuple[Dict[str, Any], int]]:
    """(slot_opened event, criterion index) pairs for every criterion an event
    satisfies. Unmatched events are simply not returned — the workflow logs
    them, nothing more. `criteria` must already be filtered to the fetch
    unit's party size (indices are into that same list)."""
    matched = []
    for ev in events:
        if ev.get("type") != "slot_opened":
            continue
        for idx, c in enumerate(criteria):
            if c.matches(ev.get("date"), ev.get("time")):
                matched.append((ev, idx))
    return matched


def relevant_open_dates(state: Dict[str, Any],
                        criteria: List[WatchCriterion]) -> List[str]:
    """Open dates in a snapshot that fall inside any active criterion's range
    (used for the one-shot calendar_published summary)."""
    return [d for d in open_dates(state)
            if any(c.status == "active" and c.date_start <= d <= c.date_end
                   for c in criteria)]


def dedupe_key(party_size: int, slot_date: Optional[str],
               slot_time: Optional[str]) -> str:
    """Notification-dedupe identity of an opened slot. Venue-level (party ×
    date × time), not per-criterion — the same opening is one piece of news
    however many criteria it satisfies."""
    return f"{party_size}|{slot_date or '-'}|{slot_time or '-'}"


# ------------------------------------------------- conversational spec parsing

_RANGE_SEP_RE = re.compile(r"\s*(?:\bto\b|\bthrough\b|\bthru\b|\buntil\b|[-–—])\s*", re.I)
_ORDINAL_RE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)\b", re.I)
_BARE_DAY_RE = re.compile(r"^\d{1,2}$")
_TIME_TOKEN_RE = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b|\b\d{1,2}:\d{2}\b", re.I)
_ANY_TIME_RE = re.compile(r"\bany\s*(?:time|slot)?\b|\bwhenever\b|\bdoesn'?t matter\b", re.I)


def parse_date_range(text: str, ctx: Optional[NormalizeCtx] = None
                     ) -> Optional[Tuple[str, str]]:
    """'September 5 to 6', 'Sept 4–12', '9/5', 'next Friday' → (start, end)
    ISO pair (start == end for a single date). None when unparseable."""
    if not text or not text.strip():
        return None
    ctx = ctx or NormalizeCtx.fresh()
    s = _ORDINAL_RE.sub(r"\1", text.strip())

    parts = _RANGE_SEP_RE.split(s, maxsplit=1)
    if len(parts) == 2:
        left = normalize_date(parts[0], ctx)
        if left is None:
            return None
        right_raw = parts[1].strip()
        if _BARE_DAY_RE.match(right_raw):
            # "Sept 4–12": the right side borrows the left's year and month.
            right = f"{left[:8]}{int(right_raw):02d}"
            try:
                date.fromisoformat(right)
            except ValueError:
                return None
        else:
            right = normalize_date(right_raw, ctx)
        if right is None or right < left:
            return None
        return left, right

    single = normalize_date(s, ctx)
    return (single, single) if single else None


def parse_times(text: str, ctx: Optional[NormalizeCtx] = None, *,
                bare: bool = False) -> Optional[List[str]]:
    """Acceptable times from free text: '7 or 9pm' → ['19:00', '21:00'];
    'any time' (or no time stated) → None, meaning any slot counts.

    Only tokens with an explicit meridiem or minutes count as times — a party
    size or a date's day never becomes a time. `bare=True` relaxes that for a
    direct answer to a "what times?" question ('7' → '19:00').
    """
    if not text or _ANY_TIME_RE.search(text):
        return None
    ctx = ctx or NormalizeCtx.fresh()
    tokens = _TIME_TOKEN_RE.findall(text)
    # "7 or 9pm": a bare number linked by "or"/"," to a meridiem-bearing time
    # shares its meridiem ("5th or 6th" never matches — ordinals aren't bare).
    linked = re.findall(r"\b(\d{1,2})\s*(?:,|or)\s+(?=\d{1,2}(?::\d{2})?\s*[ap]\.?m)",
                        text, flags=re.I)
    if not tokens and not linked:
        if not bare:
            return None
        single = normalize_time(text, ctx)
        return [single] if single else None
    meridiem = ""
    for tok in reversed(tokens):
        m = re.search(r"(a\.?m\.?|p\.?m\.?)\s*$", tok, re.I)
        if m:
            meridiem = m.group(1)
            break
    out = []
    for tok in linked + tokens:
        if meridiem and not re.search(r"[ap]\.?m\.?", tok, re.I):
            tok = f"{tok} {meridiem}"
        norm = normalize_time(tok, ctx)
        if norm and norm not in out:
            out.append(norm)
    return sorted(out) or None
