"""
BusinessDiscovery (M1).

Resolves a business and classifies *how* it takes reservations:
  - Yelp Fusion API (structured: name, phone, url, address) as the primary source.
  - Web search (Tavily, reused from Friday's search module) to find OpenTable / Resy /
    appointment-platform booking links.
  - A heuristic classifier maps the evidence to a ReservationMethod.

Both data sources are injectable so the workflow is testable without network access.
An LLM classifier (workflows/reservations/llm.py) refines the result when an
ANTHROPIC_API_KEY is available — including email-only detection — with the
deterministic heuristic below as the offline fallback. Platform-domain matches
(OpenTable/Resy) are high-precision and always win over LLM disagreement.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Tuple

from .llm import ReservationLLM, classify_method
from .models import ChannelDecision, ReservationMethod

logger = logging.getLogger(__name__)

# Booking-platform domain fragment -> reservation method.
_PLATFORM_DOMAINS = [
    ("opentable.", ReservationMethod.OPENTABLE),
    ("resy.com", ReservationMethod.RESY),
    ("vagaro.", ReservationMethod.GENERIC_WEB),
    ("booksy.", ReservationMethod.GENERIC_WEB),
    ("squareup.", ReservationMethod.GENERIC_WEB),
    ("square.site", ReservationMethod.GENERIC_WEB),
    ("mindbody", ReservationMethod.GENERIC_WEB),
    ("fresha.", ReservationMethod.GENERIC_WEB),
    ("sevenrooms.", ReservationMethod.GENERIC_WEB),
    ("exploretock.", ReservationMethod.GENERIC_WEB),
    ("tock.", ReservationMethod.GENERIC_WEB),
]

_CARD_HINT_WORDS = ("credit card", "deposit", "card to hold", "card on file", "prepay")

_EMAIL_ADDR_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Phrases that mark email as *the* way to book, not just a listed contact.
_EMAIL_BOOKING_RE = re.compile(
    r"email\s+(?:us|your request|[\w.+-]+@[\w-]+\.[\w.-]+)\s*(?:at\s+[\w.+-]+@[\w-]+\.[\w.-]+)?"
    r"\s*(?:to|for)\s+(?:book|reserve|request|make)|"
    r"reservations?\s+(?:by|via|through)\s+email|"
    r"(?:book|reserve)\s+(?:by|via|through)\s+email",
    re.I,
)


class YelpClient:
    """Minimal Yelp Fusion business-match client. Returns None on any failure."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def match(self, name: str, location: str = "") -> Optional[dict]:
        try:
            import requests

            resp = requests.get(
                "https://api.yelp.com/v3/businesses/search",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={"term": name, "location": location or "", "limit": 1},
                timeout=8,
            )
            resp.raise_for_status()
            businesses = resp.json().get("businesses", [])
            return businesses[0] if businesses else None
        except Exception:
            logger.warning("Yelp lookup failed for %r", name, exc_info=True)
            return None


class BusinessDiscovery:
    def __init__(self, search_provider=None, yelp_client=None, llm=None):
        self.search = search_provider
        self.yelp = yelp_client
        self.llm = llm

    @classmethod
    def from_env(cls) -> "BusinessDiscovery":
        yelp_key = os.getenv("YELP_API_KEY")
        yelp = YelpClient(yelp_key) if yelp_key else None

        search = None
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            try:
                from search import TavilySearchProvider
                search = TavilySearchProvider(api_key=tavily_key)
            except Exception:
                logger.warning("Could not initialise Tavily search for discovery", exc_info=True)

        return cls(search_provider=search, yelp_client=yelp, llm=ReservationLLM.from_env())

    # ------------------------------------------------------------------ discover
    def discover(self, business_name: str, location: Optional[str] = None) -> ChannelDecision:
        decision = ChannelDecision(business_name=business_name)

        # 1) Structured lookup via Yelp.
        biz = self.yelp.match(business_name, location or "") if self.yelp else None
        if biz:
            decision.business_name = biz.get("name", business_name)
            decision.phone = biz.get("phone") or biz.get("display_phone") or None
            decision.url = biz.get("url")
            addr = (biz.get("location") or {}).get("display_address")
            if addr:
                decision.address = ", ".join(addr)
            decision.source = "yelp"

        # 2) Find a booking link via web search.
        results = []
        if self.search is not None:
            try:
                results = self.search.search(f"{business_name} reservations booking", max_results=5)
            except Exception:
                logger.warning("Discovery web search failed", exc_info=True)

        booking_url, method = self._classify_links(results)
        decision.requires_card_hint = self._card_hint(results)

        if method is not None:
            decision.method = method
            decision.url = booking_url or decision.url
            decision.confidence = 0.8
            decision.source = (decision.source + "+web").strip("+")
        elif (email := self._email_booking(results)) is not None:
            # The business asks to be emailed → email channel.
            decision.method = ReservationMethod.EMAIL
            decision.email = email
            decision.confidence = 0.6
        elif decision.phone:
            # Online booking not found, but we have a number → reserve by phone.
            decision.method = ReservationMethod.PHONE
            decision.confidence = 0.5
        else:
            decision.method = ReservationMethod.UNKNOWN
            decision.confidence = 0.0

        return self._refine_with_llm(decision, biz, results)

    def _refine_with_llm(self, decision: ChannelDecision, biz: Optional[dict],
                         results) -> ChannelDecision:
        """LLM second opinion over the gathered evidence. The deterministic
        OpenTable/Resy domain match is high-precision and is never overridden;
        everything else (incl. email-only detection) defers to a valid LLM verdict.
        Evidence holds only business data — never user PII (spec §8 L5)."""
        if self.llm is None:
            return decision

        evidence = {
            "business": {
                "name": decision.business_name,
                "phone": decision.phone,
                "url": decision.url,
                "address": decision.address,
            } if biz or decision.phone or decision.url else None,
            "search_results": [
                {"title": getattr(r, "title", ""), "url": getattr(r, "url", ""),
                 "snippet": getattr(r, "snippet", "")}
                for r in (results or [])
            ],
        }
        verdict = classify_method(self.llm, evidence)
        if verdict is None:
            return decision
        if decision.method in (ReservationMethod.OPENTABLE, ReservationMethod.RESY):
            # Keep the domain match; still adopt the card hint and notes.
            decision.requires_card_hint = decision.requires_card_hint or verdict["requires_card"]
            decision.notes = verdict["notes"]
            return decision

        decision.method = ReservationMethod(verdict["method"])
        decision.url = verdict["url"] or decision.url
        decision.email = verdict["email"] or decision.email
        decision.requires_card_hint = decision.requires_card_hint or verdict["requires_card"]
        decision.confidence = verdict["confidence"]
        decision.notes = verdict["notes"]
        decision.source = (decision.source + "+llm").strip("+")
        return decision

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _classify_links(results) -> Tuple[Optional[str], Optional[ReservationMethod]]:
        """Return (url, method) for the first recognised booking platform, by priority."""
        best: Optional[Tuple[str, ReservationMethod]] = None
        for r in results or []:
            url = (getattr(r, "url", "") or "").lower()
            for fragment, method in _PLATFORM_DOMAINS:
                if fragment in url:
                    # OpenTable/Resy (dedicated reservation platforms) win over generic web.
                    if method in (ReservationMethod.OPENTABLE, ReservationMethod.RESY):
                        return getattr(r, "url", url), method
                    if best is None:
                        best = (getattr(r, "url", url), method)
        if best is not None:
            return best
        return None, None

    @staticmethod
    def _email_booking(results) -> Optional[str]:
        """An email address when the evidence says booking happens *by email*."""
        for r in results or []:
            text = f"{getattr(r, 'title', '')} {getattr(r, 'snippet', '')}"
            if _EMAIL_BOOKING_RE.search(text):
                m = _EMAIL_ADDR_RE.search(text)
                if m:
                    return m.group(0)
        return None

    @staticmethod
    def _card_hint(results) -> bool:
        for r in results or []:
            text = f"{getattr(r, 'title', '')} {getattr(r, 'snippet', '')}".lower()
            if any(w in text for w in _CARD_HINT_WORDS):
                return True
        return False
