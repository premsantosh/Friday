"""
BusinessDiscovery (M1).

Resolves a business and classifies *how* it takes reservations:
  - A structured business resolver (name, phone, url, address, hours) as the
    primary source — Google Places (preferred; generous free tier) or Yelp
    Fusion, whichever key is configured.
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

from core.harness import Sink, SinkMode, guard

from .llm import ReservationLLM, classify_method, extract_release_policy
from .models import ChannelDecision, ReleasePolicy, ReservationMethod

logger = logging.getLogger(__name__)

# Discovery queries carry ONLY business identity — never the user's name,
# phone, email, or card (spec §8 L5). Enforced, not just promised.
SEARCH_SINK = Sink("search", SinkMode.ALLOWLIST,
                   frozenset({"business_name", "location"}))

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

# For a dedicated-platform method, the domain fragment its venue page lives on
# plus the path fragments that mark a *bookable venue* (vs. a blog/cuisine/index
# page). Used to recover the venue URL when the generic search missed it.
_PLATFORM_VENUE = {
    ReservationMethod.OPENTABLE: ("opentable.", ("/r/", "/restaurant/", "/booking/")),
    ReservationMethod.RESY: ("resy.com", ("/cities/", "/venues/")),
}

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

    source_name = "yelp"

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


class GooglePlacesClient:
    """Business resolution via Google Places API (New) Text Search.

    Returns the same shape discovery reads from Yelp ({name, phone, url,
    location.display_address}) plus Yelp-style `hours`, so the two resolvers
    are interchangeable. The field mask keeps the request inside the
    Essentials/Pro SKUs (free monthly tier covers personal use). Injectable
    poster for tests; returns None on any failure.
    """

    source_name = "google_places"

    _ENDPOINT = "https://places.googleapis.com/v1/places:searchText"
    _FIELD_MASK = ",".join((
        "places.displayName",
        "places.formattedAddress",
        "places.internationalPhoneNumber",
        "places.nationalPhoneNumber",
        "places.websiteUri",
        "places.regularOpeningHours",
    ))

    def __init__(self, api_key: str, poster=None):
        self.api_key = api_key
        self._poster = poster   # callable(url, headers=..., json=..., timeout=...) → response

    def match(self, name: str, location: str = "") -> Optional[dict]:
        try:
            poster = self._poster
            if poster is None:
                import requests
                poster = requests.post
            resp = poster(
                self._ENDPOINT,
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": self.api_key,
                    "X-Goog-FieldMask": self._FIELD_MASK,
                },
                json={"textQuery": f"{name} {location}".strip(), "maxResultCount": 1},
                timeout=8,
            )
            resp.raise_for_status()
            places = resp.json().get("places") or []
            return self._normalize(places[0]) if places else None
        except Exception:
            logger.warning("Google Places lookup failed for %r", name, exc_info=True)
            return None

    @classmethod
    def _normalize(cls, place: dict) -> dict:
        from core.harness import normalize_phone

        raw_phone = (place.get("internationalPhoneNumber")
                     or place.get("nationalPhoneNumber") or "")
        address = place.get("formattedAddress")
        return {
            "name": (place.get("displayName") or {}).get("text"),
            "phone": normalize_phone(raw_phone) or (raw_phone or None),
            "url": place.get("websiteUri"),
            "location": {"display_address": [address] if address else []},
            "hours": cls._to_yelp_hours(place.get("regularOpeningHours")),
        }

    @staticmethod
    def _to_yelp_hours(opening) -> Optional[list]:
        """Google periods (day 0 = Sunday) → Yelp-style hours (day 0 = Monday),
        the format PhoneChannel's off-hours deferral already parses."""
        periods = (opening or {}).get("periods") or []
        opens = []
        for p in periods:
            o = p.get("open") or {}
            if "day" not in o:
                continue
            c = p.get("close") or {}
            start = f"{int(o.get('hour', 0)):02d}{int(o.get('minute', 0)):02d}"
            # No close (open 24h) → run to end of day; overnight closes are
            # clamped to midnight by the hours parser anyway.
            end = (f"{int(c['hour']):02d}{int(c.get('minute', 0)):02d}"
                   if "hour" in c else "2400")
            opens.append({"day": (int(o["day"]) - 1) % 7, "start": start, "end": end})
        return [{"open": opens}] if opens else None


class BusinessDiscovery:
    def __init__(self, search_provider=None, business_client=None, llm=None,
                 yelp_client=None):
        self.search = search_provider
        # `yelp_client` kept as an alias for the original keyword; any object
        # with .match(name, location) -> dict works.
        self.business = business_client if business_client is not None else yelp_client
        self.llm = llm

    @classmethod
    def from_env(cls) -> "BusinessDiscovery":
        # Prefer Google Places (free tier) when configured; Yelp Fusion otherwise.
        google_key = os.getenv("GOOGLE_PLACES_API_KEY")
        yelp_key = os.getenv("YELP_API_KEY")
        if google_key:
            business = GooglePlacesClient(google_key)
        elif yelp_key:
            business = YelpClient(yelp_key)
        else:
            business = None

        search = None
        tavily_key = os.getenv("TAVILY_API_KEY")
        if tavily_key:
            try:
                from search import TavilySearchProvider
                search = TavilySearchProvider(api_key=tavily_key)
            except Exception:
                logger.warning("Could not initialise Tavily search for discovery", exc_info=True)

        return cls(search_provider=search, business_client=business,
                   llm=ReservationLLM.from_env())

    # ------------------------------------------------------------------ discover
    def discover(self, business_name: str, location: Optional[str] = None,
                 target_url: Optional[str] = None,
                 kind: str = "dining") -> ChannelDecision:
        # The user pointed us at a page: that beats anything a search would
        # turn up, and skips two API round trips for a business that isn't in
        # a restaurant directory anyway.
        if target_url:
            return ChannelDecision(
                method=ReservationMethod.GENERIC_WEB, business_name=business_name,
                url=target_url, confidence=0.95, source="user_url",
                notes="Booking page supplied by the user.")

        # Both lookups (Yelp + web search) are built from exactly these fields.
        guard(SEARCH_SINK, {"business_name": business_name, "location": location or ""})
        decision = ChannelDecision(business_name=business_name)

        # 1) Structured lookup (Google Places or Yelp — same shape).
        biz = self.business.match(business_name, location or "") if self.business else None
        if biz:
            decision.business_name = biz.get("name", business_name)
            decision.phone = biz.get("phone") or biz.get("display_phone") or None
            decision.url = biz.get("url")
            addr = (biz.get("location") or {}).get("display_address")
            if addr:
                decision.address = ", ".join(addr)
            decision.hours = biz.get("hours")   # drives off-hours call deferral
            decision.source = getattr(self.business, "source_name", "business")

        # 2) Find a booking link via web search. Include the city when we have
        #    it — without it, common restaurant names resolve to the wrong
        #    location (e.g. "Flores" returns a New York venue instead of SF).
        results = []
        if self.search is not None:
            # A clinic doesn't call it a "reservation" — searching for the wrong
            # word buries the booking page under review-site noise.
            intent = "reservations booking" if kind == "dining" else "book appointment online"
            query = f"{business_name} {location or ''} {intent}".replace("  ", " ").strip()
            try:
                results = self.search.search(query, max_results=5)
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

        decision = self._refine_with_llm(decision, biz, results)
        return self._resolve_platform_url(decision, business_name, location)

    # ------------------------------------------------------- release policy (snipe)
    def resolve_release_policy(self, business_name: str,
                               location: Optional[str] = None) -> Optional[ReleasePolicy]:
        """Discover when a venue releases reservations, for scheduled sniping.

        Runs a general web search plus a Reddit-scoped one (forums carry the
        exact drop *time* the venue's own page usually omits), then grounds an
        LLM extraction over the combined, source-tagged evidence. Queries carry
        only business identity (spec §8 L5). Returns None when nothing usable is
        found or the LLM isn't configured."""
        if self.search is None or self.llm is None:
            return None
        guard(SEARCH_SINK, {"business_name": business_name, "location": location or ""})
        loc = (location or "").strip()
        general_q = f"{business_name} {loc} when do reservations open how far in advance".replace(
            "  ", " ").strip()
        reddit_q = f"{business_name} {loc} reservation drop time".replace("  ", " ").strip()

        results = []
        try:
            results += self.search.search(general_q, max_results=5)
        except Exception:
            logger.warning("Release-policy general search failed", exc_info=True)
        try:
            results += self.search.search(reddit_q, max_results=5,
                                          include_domains=["reddit.com"])
        except Exception:
            logger.warning("Release-policy Reddit search failed", exc_info=True)
        if not results:
            return None

        evidence = {
            "business": {"name": business_name, "location": location},
            "search_results": [
                {"source": "reddit" if "reddit.com" in (getattr(r, "url", "") or "").lower()
                          else "web",
                 "title": getattr(r, "title", ""), "url": getattr(r, "url", ""),
                 "snippet": getattr(r, "snippet", ""),
                 "published_date": getattr(r, "published_date", None)}
                for r in results
            ],
        }
        verdict = extract_release_policy(self.llm, evidence)
        if verdict is None:
            return None
        try:
            return ReleasePolicy.from_extraction(verdict)
        except (KeyError, ValueError, TypeError):
            logger.warning("Release-policy extraction returned an unusable shape", exc_info=True)
            return None

    def _resolve_platform_url(self, decision: ChannelDecision, business_name: str,
                              location: Optional[str]) -> ChannelDecision:
        """Recover the venue URL for a dedicated-platform booking.

        The method can be OpenTable/Resy (LLM or domain match) while `url` still
        points at the restaurant's own site — the generic search ranks the venue
        page below the top results often enough to miss it. The booking channels
        need the *platform* venue page (e.g. opentable.com/r/<slug>), so run one
        focused, identity-only search to find it. Carries no user PII (§8 L5)."""
        venue = _PLATFORM_VENUE.get(decision.method)
        if venue is None or self.search is None:
            return decision
        domain, path_marks = venue
        if decision.url and domain in decision.url.lower():
            return decision  # already a platform venue URL

        guard(SEARCH_SINK, {"business_name": business_name, "location": location or ""})
        query = f"{business_name} {location or ''} {decision.method.value}".strip()
        try:
            results = self.search.search(query, max_results=6)
        except Exception:
            logger.warning("Platform-URL recovery search failed", exc_info=True)
            return decision

        for r in results or []:
            url = getattr(r, "url", "") or ""
            low = url.lower()
            if domain in low and any(m in low for m in path_marks):
                decision.url = url
                decision.source = (decision.source + "+venue").strip("+")
                return decision
        return decision

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
