"""
Reservation data model (M1).

Slots live in the session's `slots` dict (the multi-turn framework owns that);
these types are for the structured results that flow between the workflow, the
discovery step, and (later) the channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ReservationMethod(Enum):
    OPENTABLE = "opentable"
    RESY = "resy"
    YELP = "yelp"
    GENERIC_WEB = "generic_web"
    PHONE = "phone"
    EMAIL = "email"
    UNKNOWN = "unknown"


# Essentials collected up front, each with its deterministic normalizer —
# values are canonicalized (ISO date, 24h time, bounded int) the moment they
# enter the slots, and re-asked when the normalizer rejects them.
# Channel-specific extras (email, card, seating preference, …) are gathered
# lazily once a channel is chosen.
from core.harness import (  # noqa: E402  (import after docstring block by design)
    SlotSpec,
    normalize_date,
    normalize_party_size,
    normalize_text,
    normalize_time,
)

ESSENTIAL_SLOT_SPECS: tuple = (
    SlotSpec("business_name",
             "Which establishment shall I book, sir?",
             "I didn't catch the establishment's name, sir — which place shall I book?",
             normalize_text),
    SlotSpec("date",
             "For which date, sir?",
             "I couldn't make out the date, sir — something like \"next Friday\" or \"June 14\"?",
             normalize_date),
    SlotSpec("time",
             "At what time, sir?",
             "I couldn't make out the time, sir — something like \"7pm\"?",
             normalize_time),
    SlotSpec("party_size",
             "For how many people, sir?",
             "How many shall I say, sir — just a number?",
             normalize_party_size),
    SlotSpec("service_type",
             "What shall I book it for, sir — what sort of appointment?",
             "I didn't catch what the appointment is for, sir — a consultation, "
             "a haircut, something else?",
             normalize_text),
)

SLOT_SPECS_BY_NAME: Dict[str, SlotSpec] = {s.name: s for s in ESSENTIAL_SLOT_SPECS}
SLOT_PROMPTS: Dict[str, str] = {s.name: s.prompt for s in ESSENTIAL_SLOT_SPECS}


class BookingKind(Enum):
    """What sort of booking this is — it decides which slots are essential.

    A restaurant needs a party size and an exact time; a physiotherapy
    consultation needs neither, and asking "for how many people?" before a
    clinic appointment is how this workflow used to give itself away.
    """
    DINING = "dining"            # a table: business, date, time, party size
    APPOINTMENT = "appointment"  # a service at a time: business, service, date
    INQUIRY = "inquiry"          # a request form with no times offered


_REQUIRED_BY_KIND: Dict[BookingKind, tuple] = {
    BookingKind.DINING: ("business_name", "date", "time", "party_size"),
    BookingKind.APPOINTMENT: ("business_name", "service_type", "date"),
    BookingKind.INQUIRY: ("business_name", "service_type"),
}

# The historical name, still the dining set — imported by workflows/__init__.py.
ESSENTIAL_SLOTS: tuple = _REQUIRED_BY_KIND[BookingKind.DINING]


def parse_kind(value: Any, default: BookingKind = BookingKind.DINING) -> BookingKind:
    """Tolerant parse of a kind from a slot or an LLM field."""
    if isinstance(value, BookingKind):
        return value
    try:
        return BookingKind(str(value).strip().lower())
    except (ValueError, AttributeError):
        return default


def required_slots(kind: Any) -> tuple:
    """The slots that must be filled before this kind of booking can proceed."""
    return _REQUIRED_BY_KIND[parse_kind(kind)]

_METHOD_PHRASES: Dict[ReservationMethod, str] = {
    ReservationMethod.OPENTABLE: "They take reservations through OpenTable.",
    ReservationMethod.RESY: "They use Resy for reservations.",
    ReservationMethod.YELP: "They accept reservations via Yelp.",
    ReservationMethod.GENERIC_WEB: "They have an online booking page.",
    ReservationMethod.PHONE: "I found no online booking, so I'd reserve by telephone.",
    ReservationMethod.EMAIL: "They take reservation requests by email.",
    ReservationMethod.UNKNOWN: "I could not yet determine how they take reservations.",
}


@dataclass
class ReleasePolicy:
    """When a venue releases reservations, as discovered from the web/forums."""
    days_in_advance: int
    release_time: str               # verbatim clock time, e.g. "10am"
    timezone: Optional[str] = None  # "ET" / "America/New_York" / None
    rolling: bool = True
    confidence: float = 0.0
    source_quote: str = ""
    notes: str = ""

    @classmethod
    def from_extraction(cls, d: Dict[str, Any]) -> "ReleasePolicy":
        return cls(
            days_in_advance=int(d["opens_days_in_advance"]),
            release_time=d["release_time"],
            timezone=d.get("release_timezone"),
            rolling=bool(d.get("rolling", True)),
            confidence=float(d.get("confidence", 0.0)),
            source_quote=d.get("source_quote", ""),
            notes=d.get("notes", ""),
        )


@dataclass
class ChannelDecision:
    """How a business takes reservations, plus the contact details discovery found."""
    method: ReservationMethod = ReservationMethod.UNKNOWN
    business_name: Optional[str] = None
    url: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    hours: Optional[Any] = None
    timezone: Optional[str] = None
    confidence: float = 0.0
    requires_card_hint: bool = False
    notes: str = ""
    source: str = ""

    def method_phrase(self) -> str:
        return _METHOD_PHRASES.get(self.method, _METHOD_PHRASES[ReservationMethod.UNKNOWN])

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form for persisting into session.slots."""
        return {
            "method": self.method.value,
            "business_name": self.business_name,
            "url": self.url,
            "phone": self.phone,
            "email": self.email,
            "address": self.address,
            "hours": self.hours,
            "timezone": self.timezone,
            "confidence": self.confidence,
            "requires_card_hint": self.requires_card_hint,
            "notes": self.notes,
            "source": self.source,
        }
