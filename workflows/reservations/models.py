"""
Reservation data model (M1).

Slots live in the session's `slots` dict (the multi-turn framework owns that);
these types are for the structured results that flow between the workflow, the
discovery step, and (later) the channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ReservationMethod(Enum):
    OPENTABLE = "opentable"
    RESY = "resy"
    YELP = "yelp"
    GENERIC_WEB = "generic_web"
    PHONE = "phone"
    EMAIL = "email"
    UNKNOWN = "unknown"


# Essentials collected up front. Channel-specific extras (email, card, seating
# preference, …) are gathered lazily once a channel is chosen.
ESSENTIAL_SLOTS: tuple = ("business_name", "date", "time", "party_size")

SLOT_PROMPTS: Dict[str, str] = {
    "business_name": "Which establishment shall I book, sir?",
    "date": "For which date, sir?",
    "time": "At what time, sir?",
    "party_size": "For how many people, sir?",
}

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
            "timezone": self.timezone,
            "confidence": self.confidence,
            "requires_card_hint": self.requires_card_hint,
            "notes": self.notes,
            "source": self.source,
        }
