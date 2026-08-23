"""
TableCheckChannel (M8/M9) — availability polling **and booking** for venues on
tablecheck.com (e.g. https://www.tablecheck.com/en/benfiddich-tokyo/reserve).

Pure HTTP against the same JSON API the reserve widget uses (verified live
against the widget's own XHR traffic on 2026-08-19; see docs/… M8/M9 notes):

  POST /booking/calendar        day-level state for a (party, date range):
                                {"calendar": {"2026-09-09": [], ...}}. An empty
                                flag list means bookable; "closed" means the
                                venue is shut / hasn't released that day yet;
                                "unavailable" / "cache_unavailable" (in the
                                widget's strict mode) mean full for that party.
  POST /booking/meals_v2        per-time availability for one day:
                                {"meals": {"all_day": [{"t": "<UTC ISO>", "a": bool}]}}
  GET  /booking/booking_pages   venue settings (tz, service categories,
                                required questions, payment mode).
  POST /booking/menu_items      the "experience"/menu item a booking must carry.
  POST /booking/cart            creates a cart = a 10-minute HOLD on the slot,
                                and validates the customer/order/answers body.
  PUT  /booking/checkout/:cart  submits the card token (GMO) and books.
  PUT  /booking/checkout_status/:cart   poll: reservation | pending | 3DS redirect.
  DELETE /booking/cart/:cart    releases a hold we don't intend to complete.

`fetch_month` and `fetch_day_slots` are the polling primitives (cheap, no
side effects). `commit` performs the booking; the only side-effecting call
before checkout is the cart (a transient hold), which is released on every
failure path except a pending 3-D Secure challenge that the user may still
finish. Card data reaches only GMO's tokeniser (in a throwaway headless page)
and never the API body, logs, or the LLM.

Response parsing is deliberately strict: anything off-shape raises
SchemaDrift so the watcher pauses and alerts instead of silently reporting
"no availability" (drift is the #1 failure mode).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import urllib.parse
import uuid
from calendar import monthrange
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from ..models import ChannelDecision, ReservationMethod
from .base import (
    Availability,
    AvailabilityStatus,
    BookingResult,
    CommitPlan,
    ReservationChannel,
)

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://production-booking.tablecheck.com/v2"
GMO_TOKEN_JS = "https://static.mul-pay.jp/ext/js/token.js"

# Browser-equivalent headers: this is the same traffic the widget sends, at a
# far lower rate than a human clicking through months (ToS etiquette).
_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "Accept-Language": "en",
    "Origin": "https://www.tablecheck.com",
    "Referer": "https://www.tablecheck.com/",
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
}

# Calendar flags. Absent flags == bookable for that party size.
_CLOSED_FLAGS = frozenset({"closed"})
_FULL_FLAGS = frozenset({"unavailable", "cache_unavailable"})
_KNOWN_FLAGS = _CLOSED_FLAGS | _FULL_FLAGS | {"res_request_available"}

_SLUG_RE = re.compile(
    r"tablecheck\.com/(?:[a-z]{2}(?:-[a-z]{2})?/)?(?:shops?/)?([a-z0-9][a-z0-9\-_]+)",
    re.I,
)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Availability error codes from the cart endpoint that mean "not this slot,
# right now" (worth retrying another slot / next tick) vs "not open yet".
_NO_SLOT_CODES = frozenset({"no_availability", "failure", "unavailable"})
_NOT_OPEN_CODES = frozenset({"max_time_cutoff"})


class TableCheckError(Exception):
    """HTTP-level failure (network error, 429, 5xx…). `status` is the HTTP
    status code when there was a response, else None."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class SchemaDrift(TableCheckError):
    """The response parsed as JSON but didn't match the known shape — the
    endpoint has likely changed. The watcher pauses + alerts on this rather
    than treating it as 'no availability' (drift is the #1 failure mode)."""


def slug_from_url(url: Optional[str]) -> Optional[str]:
    """'https://www.tablecheck.com/en/benfiddich-tokyo/reserve' → 'benfiddich-tokyo'."""
    if not url:
        return None
    m = _SLUG_RE.search(url)
    return m.group(1) if m else None


def _tr(translations: Any, locale: str = "en") -> str:
    """Pick one translation from TableCheck's [{translation, locale}] lists."""
    if not isinstance(translations, list):
        return ""
    for t in translations:
        if isinstance(t, dict) and t.get("locale") == locale and t.get("translation"):
            return str(t["translation"])
    for t in translations:
        if isinstance(t, dict) and t.get("translation"):
            return str(t["translation"])
    return ""


def _e164(phone: str, default_cc: str = "1") -> str:
    """'650-555-0100' → '+16505550100'; an existing +country code is kept.
    TableCheck validates phones strictly (E.164), and the profile stores them
    the way a person types them."""
    raw = (phone or "").strip()
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return ""
    if raw.startswith("+"):
        return "+" + digits
    if raw.startswith("00") and len(digits) > 11:
        return "+" + digits[2:]
    if len(digits) == 10:
        return f"+{default_cc}{digits}"
    if len(digits) == 11 and digits.startswith(default_cc):
        return "+" + digits
    return "+" + digits


def _split_name(full: str) -> Tuple[str, str]:
    parts = (full or "").strip().split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], parts[0]
    return " ".join(parts[:-1]), parts[-1]


class TableCheckChannel(ReservationChannel):
    method = ReservationMethod.TABLECHECK
    can_commit = True

    def __init__(self, fetcher: Optional[Callable] = None,
                 tokenizer: Optional[Callable] = None):
        # fetcher: async callable(method, url, headers, json_body) -> (status, json_or_None).
        # tokenizer: async callable(public_key, card, holder) -> gmo token string.
        # Both injectable for tests; the defaults use aiohttp / headless Chromium.
        self._fetcher = fetcher
        self._tokenizer = tokenizer
        self.api_base = os.getenv("TABLECHECK_API_BASE", DEFAULT_API_BASE).rstrip("/")
        self._venue_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ polling
    async def fetch_month(self, slug: str, party_size: int, month: str) -> Dict[str, Any]:
        """Normalized day-level availability snapshot for one (venue, party
        size, month) fetch unit — the primitive the watcher polls.

        Raises TableCheckError on HTTP/network trouble, SchemaDrift when the
        response shape isn't recognised.
        """
        year, mon = (int(x) for x in month.split("-"))
        last_day = monthrange(year, mon)[1]
        tz = await self._venue_tz(slug)
        body = {
            "shop_id": slug,
            "start_date": f"{month}-01T00:00:00.000{tz}",
            "end_date": f"{month}-{last_day:02d}T23:59:59.999{tz}",
            "bookable_menu_list_ids": [], "bookable_menu_item_ids": [],
            "voucher_ids": [], "menu_item_ids": [], "unavailable": True,
            "pax_adult": int(party_size), "pax_senior": 0, "pax_child": 0, "pax_baby": 0,
        }
        status, raw = await self._call("POST", "/booking/calendar", body)
        self._raise_for_status(status, "calendar")
        return self._parse_raw(raw, month)

    async def fetch_day_slots(self, slug: str, party_size: int, date_iso: str
                              ) -> Dict[str, str]:
        """{"HH:MM" (venue-local): "available"|"full"} for one day."""
        body = {"shop_id": slug, "date": date_iso, "pax_adult": int(party_size),
                "pax_senior": 0, "pax_child": 0, "pax_baby": 0, "locale": "en"}
        status, raw = await self._call("POST", "/booking/meals_v2", body)
        self._raise_for_status(status, "meals")
        tzname = await self._venue_tzname(slug)
        return self._parse_meals(raw, tzname)

    # ---------------------------------------------------------------- transport
    async def _call(self, method: str, path: str, body: Optional[Dict[str, Any]] = None
                    ) -> Tuple[int, Any]:
        url = self.api_base + path
        if self._fetcher is not None:
            return await self._fetcher(method, url, _HEADERS, body)
        try:
            import aiohttp
        except ImportError as exc:
            raise TableCheckError(f"aiohttp is not installed: {exc}") from exc
        try:
            timeout = aiohttp.ClientTimeout(total=25)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.request(method, url, headers=_HEADERS,
                                        json=body if body is not None else None) as resp:
                    try:
                        data = await resp.json(content_type=None)
                    except Exception:
                        data = None
                    return resp.status, data
        except TableCheckError:
            raise
        except Exception as exc:
            raise TableCheckError(f"TableCheck request failed: {exc}") from exc

    @staticmethod
    def _raise_for_status(status: int, what: str) -> None:
        if status == 429:
            raise TableCheckError("Rate limited by TableCheck (429).", status=429)
        if status >= 500:
            raise TableCheckError(f"TableCheck server error ({status}).", status=status)
        if status != 200:
            # Unexpected 4xx is drift territory: the endpoint moved or the slug
            # scheme changed — don't keep hammering it.
            raise SchemaDrift(f"Unexpected HTTP {status} from {what} endpoint.",
                              status=status)

    # ------------------------------------------------------------- venue info
    async def venue_info(self, slug: str) -> Dict[str, Any]:
        """Cached booking-page settings for a venue (tz, service categories,
        questions, payment mode, party limits). Raises on failure."""
        cached = self._venue_cache.get(slug)
        if cached is not None:
            return cached
        status, raw = await self._call(
            "GET", f"/booking/booking_pages?shop_slug={urllib.parse.quote(slug)}&locale=en")
        self._raise_for_status(status, "booking_pages")
        if not isinstance(raw, dict) or not isinstance(raw.get("shop"), dict) \
                or not isinstance(raw.get("booking_page"), dict):
            raise SchemaDrift("booking_pages response missing shop/booking_page.")
        shop, page = raw["shop"], raw["booking_page"]
        services = page.get("booking_services") or []
        service = services[0] if services and isinstance(services[0], dict) else {}
        info = {
            "shop_id": shop.get("id"),
            "slug": shop.get("slug") or slug,
            "name": _tr(shop.get("name_translations")) or slug,
            "time_zone": shop.get("time_zone") or "Asia/Tokyo",
            "tz_offset_s": shop.get("tz_utc_offset"),
            "phone": shop.get("phone"),
            "enable_payments": bool(shop.get("enable_payments")),
            "currency": shop.get("currency") or "",
            "max_num_people": page.get("max_num_people"),
            "min_num_people": page.get("min_num_people"),
            "input_service_category": page.get("input_service_category"),
            "service_categories": [
                {"id": sc.get("id"), "name": _tr(sc.get("name_translations")),
                 "description": _tr(sc.get("description_translations"))}
                for sc in (shop.get("service_categories") or []) if isinstance(sc, dict)],
            "questions": [
                {"id": q.get("id"), "required": bool(q.get("is_required")),
                 "ui_type": q.get("ui_type"),
                 "text": _tr(q.get("question_translations")),
                 "options": [o.get("id") for o in (q.get("question_options") or [])
                             if isinstance(o, dict) and o.get("id")]}
                for q in (service.get("questions") or []) if isinstance(q, dict)],
            "service_mode": service.get("service_mode") or "dining",
            "cancel_policy": _tr(service.get("cancel_policy_translations")),
            "booking_policy": _tr(service.get("merchant_message_translations")),
        }
        self._venue_cache[slug] = info
        return info

    async def _venue_tzname(self, slug: str) -> str:
        try:
            return (await self.venue_info(slug)).get("time_zone") or "Asia/Tokyo"
        except TableCheckError:
            raise
        except Exception:
            return "Asia/Tokyo"

    async def _venue_tz(self, slug: str) -> str:
        """'+09:00' style offset for calendar range bounds."""
        tzname = await self._venue_tzname(slug)
        try:
            off = datetime.now(ZoneInfo(tzname)).strftime("%z")
            return f"{off[:3]}:{off[3:]}"
        except Exception:
            return "+09:00"

    # -------------------------------------------------------------- normalize
    @staticmethod
    def _parse_raw(raw: Any, month: str) -> Dict[str, Any]:
        """/booking/calendar response → the normalized snapshot shape the
        differ consumes:
          {"month", "dates": {iso: {"open": bool, "slots": {}}}, "venue_status": "ok"}
        Closed days are *absent* (so a month flipping from all-closed to
        released reads as `calendar_published`); a present day is open when
        it carries no flags, full when flagged unavailable/cache_unavailable.
        """
        if not isinstance(raw, dict):
            raise SchemaDrift("Calendar response is not a JSON object.")
        if raw.get("errors"):
            raise SchemaDrift(f"Calendar endpoint returned errors: {raw['errors']!r}")
        calendar = raw.get("calendar")
        if not isinstance(calendar, dict):
            raise SchemaDrift("Calendar response has no 'calendar' object.")

        dates: Dict[str, Dict[str, Any]] = {}
        for day, flags in calendar.items():
            if not _DATE_RE.match(str(day)):
                raise SchemaDrift(f"Calendar key {day!r} is not a date.")
            if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
                raise SchemaDrift(f"Calendar value for {day!r} is not a flag list.")
            unknown = [f for f in flags if f not in _KNOWN_FLAGS]
            if unknown:
                raise SchemaDrift(f"Unknown calendar flag(s) {unknown!r} for {day!r}.")
            if any(f in _CLOSED_FLAGS for f in flags):
                continue
            is_open = not any(f in _FULL_FLAGS for f in flags)
            dates[day] = {"open": is_open, "slots": {}}
        return {"month": month, "dates": dates, "venue_status": "ok"}

    @staticmethod
    def _parse_meals(raw: Any, tzname: str) -> Dict[str, str]:
        if not isinstance(raw, dict) or not isinstance(raw.get("meals"), dict):
            raise SchemaDrift("meals_v2 response has no 'meals' object.")
        try:
            tz = ZoneInfo(tzname)
        except Exception:
            tz = ZoneInfo("Asia/Tokyo")
        out: Dict[str, str] = {}
        for _, entries in raw["meals"].items():
            if not isinstance(entries, list):
                raise SchemaDrift("meals_v2 meal block is not a list.")
            for e in entries:
                if not isinstance(e, dict) or "t" not in e or "a" not in e:
                    raise SchemaDrift("meals_v2 entry without 't'/'a'.")
                try:
                    dt = datetime.fromisoformat(str(e["t"]).replace("Z", "+00:00"))
                except ValueError as exc:
                    raise SchemaDrift(f"meals_v2 time {e['t']!r} unparseable.") from exc
                local = dt.astimezone(tz).strftime("%H:%M")
                out[local] = "available" if e.get("a") is True else "full"
        return out

    # ------------------------------------------------------------- deep links
    @staticmethod
    def booking_url(slug: str, date: Optional[str] = None,
                    time: Optional[str] = None,
                    party_size: Optional[int] = None) -> str:
        """Reserve-page deep link (the widget ignores unknown params, so the
        bare /reserve page always works)."""
        params = {}
        if date:
            params["start_date"] = date
        if time:
            params["start_time"] = time
        if party_size:
            params["num_people"] = str(party_size)
        query = ("?" + urllib.parse.urlencode(params)) if params else ""
        return f"https://www.tablecheck.com/en/{urllib.parse.quote(slug)}/reserve{query}"

    # ----------------------------------------------------------- channel ABC
    async def check_availability(self, slots: Dict[str, Any]) -> Availability:
        """Concrete availability for the booking flow's (date, time, party).
        UNKNOWN only when we genuinely can't tell (no slug, fetch trouble)."""
        slug = self._slug_from_slots(slots)
        date, time_hhmm = slots.get("date"), slots.get("time")
        party = slots.get("party_size")
        if not (slug and date and party):
            return Availability(status=AvailabilityStatus.UNKNOWN,
                                note="I couldn't identify the TableCheck venue page.")
        try:
            state = await self.fetch_month(slug, int(party), str(date)[:7])
        except TableCheckError as exc:
            logger.warning("TableCheck availability check failed: %s", exc)
            return Availability(status=AvailabilityStatus.UNKNOWN,
                                note="TableCheck wasn't reachable just now.")
        info = state["dates"].get(str(date)) or {}
        if not info.get("open"):
            return Availability(status=AvailabilityStatus.UNAVAILABLE)
        try:
            times = await self.fetch_day_slots(slug, int(party), str(date))
        except TableCheckError:
            return Availability(status=AvailabilityStatus.AVAILABLE)
        available = sorted(t for t, s in times.items() if s == "available")
        if not time_hhmm or time_hhmm in available:
            return Availability(status=AvailabilityStatus.AVAILABLE, options=available)
        return Availability(status=AvailabilityStatus.UNAVAILABLE, options=available)

    async def prepare(self, slots: Dict[str, Any], decision: ChannelDecision) -> CommitPlan:
        slug = self._slug_from_slots(slots) or slug_from_url(decision.url) or ""
        url = self.booking_url(slug, slots.get("date"), slots.get("time"),
                               slots.get("party_size")) if slug else decision.url
        summary = (f"Book {decision.business_name or slots.get('business_name')} via "
                   f"TableCheck for {slots.get('party_size', 'your party')} on "
                   f"{slots.get('date')} at {slots.get('time')}")
        return CommitPlan(
            channel="tablecheck", summary=summary,
            details={"url": url, "slug": slug,
                     "business_name": decision.business_name or slots.get("business_name"),
                     "date": slots.get("date"), "time": slots.get("time"),
                     "party_size": slots.get("party_size"),
                     "guest_name": slots.get("guest_name"),
                     "email": slots.get("email"), "phone": slots.get("phone")},
            # TableCheck venues that take payments register a card at booking
            # (no-show / late-cancel protection); assume so unless told otherwise.
            requires_card=True,
        )

    async def commit(self, plan: CommitPlan, payment: Any = None) -> BookingResult:
        """Book the exact (slug, date, time, party) in `plan.details`.

        Steps: venue settings → menu item → cart (validates + holds the slot
        for ~10 min) → payment (GMO card token, if the venue asks) → checkout →
        status. Every failure after the cart is created releases the hold,
        except a pending 3-D Secure challenge that the user can still finish
        from the link we return.
        """
        d = plan.details or {}
        slug, date, time_hhmm = d.get("slug"), d.get("date"), d.get("time")
        party = d.get("party_size")
        link = self.booking_url(slug, date, time_hhmm, party) if slug else d.get("url")
        if not (slug and date and time_hhmm and party):
            return BookingResult(success=False, needs_manual=True, error="incomplete_plan",
                                 message=f"I don't have a full slot to book, sir: {link}")
        try:
            return await self._book(plan, payment, link)
        except TableCheckError as exc:
            logger.warning("TableCheck booking failed: %s", exc)
            return BookingResult(
                success=False, needs_manual=True, error="tablecheck_error",
                message=f"TableCheck wasn't cooperating ({exc}), sir. Finish it here: {link}")
        except Exception as exc:  # never let a booking crash the watch tick
            logger.exception("TableCheck booking crashed")
            return BookingResult(
                success=False, needs_manual=True, error=f"crash:{type(exc).__name__}",
                message=f"Something went wrong mid-booking ({exc}), sir. Finish it here: {link}")

    async def _book(self, plan: CommitPlan, payment: Any, link: str) -> BookingResult:
        d = plan.details
        slug, date, time_hhmm, party = d["slug"], str(d["date"]), str(d["time"]), int(d["party_size"])
        info = await self.venue_info(slug)
        biz = d.get("business_name") or info.get("name") or slug
        tzname = info.get("time_zone") or "Asia/Tokyo"

        max_pax = info.get("max_num_people")
        if isinstance(max_pax, int) and party > max_pax:
            return BookingResult(
                success=False, needs_manual=True, error="party_too_large",
                message=f"{biz} takes at most {max_pax} online, sir — a party of {party} "
                        f"needs a call ({info.get('phone') or 'number on their page'}).")

        # Customer facts. Both names, email and phone are required by TableCheck.
        first, last = _split_name(d.get("guest_name") or "")
        email = (d.get("email") or "").strip()
        phone = _e164(d.get("phone") or "", os.getenv("RESERVATION_PHONE_COUNTRY_CODE", "1"))
        if not (first and last and email and phone):
            return BookingResult(
                success=False, needs_manual=True, error="missing_contact",
                message=f"I need a full name, email and phone to book {biz}, sir. "
                        f"Finish it here: {link}")

        # start_at in venue-local time.
        try:
            offset = datetime.now(ZoneInfo(tzname)).strftime("%z")
        except Exception:
            offset = "+0900"
        start_at = f"{date}T{time_hhmm}:00.000{offset[:3]}:{offset[3:]}"

        menu_item = await self._pick_menu_item(slug, date, d.get("menu_item_id"))
        orders = ([{"is_group_order": bool(menu_item.get("is_group_order")),
                    "qty": 1, "menu_item_id": menu_item["id"], "voucher_ids": []}]
                  if menu_item else [])
        service_category_id = self._pick_service_category(info, party, d.get("service_category_id"))
        answers = self._default_answers(info)

        session_ref = str(uuid.uuid4())
        if info.get("shop_id"):
            # Same session bootstrap the widget performs; harmless if ignored.
            try:
                await self._call("POST", "/booking/cart/init",
                                 {"shop_id": info["shop_id"], "locale": "en",
                                  "session_ref": session_ref})
            except TableCheckError:
                pass

        body = {
            "shop_slug": slug, "service_mode": info.get("service_mode") or "dining",
            "pax_adult": party, "pax_senior": 0, "pax_child": 0, "pax_baby": 0,
            "start_at": start_at, "manual_duration": None, "seat_types": [],
            "smoking": "none", "service_category_id": service_category_id,
            "purpose": None, "room_name": "", "visit_history": None,
            "special_request": (d.get("special_request") or "")[:500],
            "orders": orders, "answers": answers, "membership_auth_id": None,
            "allow_marketing": False, "use_experience_page": False,
            "is_smartpay_requested": False, "voucher_ids": [],
            "customer": {"first_name": first, "last_name": last, "is_single_name": False,
                         "email": email, "phone": phone},
            "send_texts": False, "locale": "en", "session_ref": session_ref,
        }
        status, raw = await self._call("POST", "/booking/cart", body)
        self._raise_for_status(status, "cart")
        if not isinstance(raw, dict):
            raise SchemaDrift("Cart response is not a JSON object.")
        cart = raw.get("cart") if isinstance(raw.get("cart"), dict) else {}
        avail = raw.get("availability") if isinstance(raw.get("availability"), dict) else None
        errors = cart.get("validation_errors") or {}
        cart_id = cart.get("id")

        if errors:
            if cart_id:
                await self._release_cart(cart_id)
            return BookingResult(
                success=False, needs_manual=True, error="validation",
                message=f"TableCheck rejected the booking details for {biz}: "
                        f"{self._flatten_errors(errors)}. Finish it here: {link}")
        if not cart_id:
            code = str((avail or {}).get("code") or (avail or {}).get("data") or "no_availability")
            msg = (avail or {}).get("message")
            if code in _NOT_OPEN_CODES:
                return BookingResult(success=False, error="not_open",
                                     message=f"{biz} isn't taking bookings for {date} yet"
                                             f"{' (' + msg + ')' if msg else ''}.")
            return BookingResult(success=False, error="slot_not_found",
                                 message=f"That {time_hhmm} table on {date} at {biz} is gone.")

        # Payment. `register` = card on file for the cancel policy, nothing charged now.
        needs_card = bool(cart.get("show_payment_form"))
        if needs_card:
            charge_now = self._to_float(cart.get("payment_total_amt"))
            txn = str(cart.get("payment_transaction_type") or "")
            if txn not in ("register", "") or charge_now > 0:
                await self._release_cart(cart_id)
                amt = f"{charge_now:,.0f} {info.get('currency', '')}".strip()
                return BookingResult(
                    success=False, needs_manual=True, error="over_cap",
                    message=f"{biz} wants a real charge up front ({txn or 'charge'} {amt}), "
                            f"which is beyond what I'm allowed to pay, sir. Finish it here: {link}")
            if payment is None:
                await self._release_cart(cart_id)
                return BookingResult(
                    success=False, needs_manual=True, error="card_required",
                    message=f"{biz} needs a card on file for its cancel policy, sir, and I "
                            f"don't have one to use. Finish it here: {link}")
            gateway = self._gmo_gateway(cart)
            if gateway is None:
                await self._release_cart(cart_id)
                return BookingResult(
                    success=False, needs_manual=True, error="unsupported_gateway",
                    message=f"{biz} uses a card processor I can't drive, sir. "
                            f"I'm holding nothing — finish it here: {link}")
            holder = f"{first} {last}".upper()
            try:
                token = await self._gmo_token(gateway["public_key"], payment, holder)
            except Exception as exc:
                await self._release_cart(cart_id)
                return BookingResult(
                    success=False, needs_manual=True, error="card_token_failed",
                    message=f"The card processor wouldn't accept the card details for {biz} "
                            f"({exc}), sir. Finish it here: {link}")
            checkout = {
                "payment_method": "card", "payment_gateway_id": gateway["id"],
                "is_smartpay_requested": False, "gateway_token": token,
                "booking_fee_gateway_token": "", "payment_profile_id": None,
                "name_on_card": holder, "card_digits": str(payment.last_four),
                "expiry_year": self._year4(payment.exp_year),
                "expiry_month": str(payment.exp_month).zfill(2),
                "card_brand": self._brand(payment.pan),
            }
        else:
            checkout = {}

        status, raw = await self._call("PUT", f"/booking/checkout/{cart_id}", checkout)
        if status >= 500 or status == 429:
            await self._release_cart(cart_id)
            self._raise_for_status(status, "checkout")
        result = self._interpret_checkout(raw, biz, link)
        if result is None:
            # Pending: poll the status endpoint for a while.
            result = await self._await_checkout(cart_id, biz, link, raw)
        if not result.success and result.error != "3ds_pending":
            await self._release_cart(cart_id)
        return result

    # ------------------------------------------------------- booking helpers
    async def _pick_menu_item(self, slug: str, date_iso: str,
                              preferred_id: Optional[str]) -> Optional[Dict[str, Any]]:
        status, raw = await self._call("POST", "/booking/menu_items",
                                       {"shop_id": slug, "start_date": date_iso,
                                        "use_experience_page": False, "locale": "en"})
        self._raise_for_status(status, "menu_items")
        items = (raw or {}).get("menu_items") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            raise SchemaDrift("menu_items response has no list.")
        items = [i for i in items if isinstance(i, dict) and i.get("id")]
        if preferred_id:
            for i in items:
                if i["id"] == preferred_id:
                    return i
        usable = [i for i in items
                  if i.get("is_active", True) and not i.get("is_hidden")
                  and i.get("availability_status", "available") == "available"]
        if not usable:
            return None
        # Prefer the venue's plain seat/table item shown to English users:
        # main-type, and multilingual (`show_locales` includes en) if such exists.
        def rank(i):
            return (0 if i.get("menu_type") == "main" else 1,
                    0 if "en" in (i.get("show_locales") or []) else 1,
                    0 if i.get("price") in (None, 0, "0", "0.0") else 1,
                    int(i.get("position") or 0))
        return sorted(usable, key=rank)[0]

    @staticmethod
    def _pick_service_category(info: Dict[str, Any], party: int,
                               preferred_id: Optional[str]) -> Optional[str]:
        cats = info.get("service_categories") or []
        if preferred_id and any(c.get("id") == preferred_id for c in cats):
            return preferred_id
        if not cats:
            return None
        # "Reservations can be made for 2 to 4 people." → pick the category
        # whose stated range covers the party; else the first one.
        for c in cats:
            nums = [int(n) for n in re.findall(r"\d+", c.get("description") or "")]
            if len(nums) >= 2 and nums[0] <= party <= nums[-1]:
                return c.get("id")
            if len(nums) == 1 and party == nums[0]:
                return c.get("id")
        for c in cats:
            if not re.search(r"\d", c.get("description") or ""):
                return c.get("id")
        return cats[-1].get("id")

    @staticmethod
    def _default_answers(info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Answer only *required* venue questions, minimally: an acknowledgment
        radio/checkbox picks its (first) option; a required free-text gets a
        neutral 'None'. Optional questions are left blank."""
        answers = []
        for q in info.get("questions") or []:
            if not q.get("required"):
                continue
            ui, qid, opts = q.get("ui_type"), q.get("id"), q.get("options") or []
            if ui == "radio" and opts:
                answers.append({"question_id": qid, "is_selected": False,
                                "question_option_id": opts[0]})
            elif ui == "checkbox":
                answers.append({"question_id": qid, "is_selected": True})
            elif ui == "checkboxes" and opts:
                answers.append({"question_id": qid, "question_option_ids": [opts[0]],
                                "is_selected": False})
            elif ui == "textarea":
                answers.append({"question_id": qid, "text": "None"})
        return answers

    @staticmethod
    def _gmo_gateway(cart: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for g in cart.get("payment_gateways") or []:
            if (isinstance(g, dict) and g.get("provider") == "gmo"
                    and "card" in (g.get("payment_methods") or ["card"])
                    and g.get("public_key") and g.get("id")):
                return g
        return None

    async def _gmo_token(self, public_key: str, card: Any, holder: str) -> str:
        """Tokenise the card with GMO's token.js in a throwaway headless page.
        The PAN/CVV go to GMO only; the token is single-use and safe to send."""
        if self._tokenizer is not None:
            return await self._tokenizer(public_key, card, holder)
        from playwright.async_api import async_playwright
        expire = f"{self._year4(card.exp_year)}{str(card.exp_month).zfill(2)}"
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto("https://www.tablecheck.com/en/", wait_until="domcontentloaded",
                                timeout=30000)
                await page.add_script_tag(url=GMO_TOKEN_JS)
                res = await page.evaluate(
                    """(args) => new Promise((resolve) => {
                        try {
                          Multipayment.init(args.key);
                          Multipayment.getToken(
                            {holdername: args.holder, expire: args.expire,
                             cardno: args.pan, securitycode: args.cvv},
                            (r) => resolve({code: r.resultCode,
                                            token: r.tokenObject && r.tokenObject.token}));
                        } catch (e) { resolve({code: 'JS', error: String(e)}); }
                    })""",
                    {"key": public_key, "holder": holder, "expire": expire,
                     "pan": re.sub(r"\D", "", str(card.pan)), "cvv": str(card.cvv)})
            finally:
                await browser.close()
        if not isinstance(res, dict) or res.get("code") != "000" or not res.get("token"):
            raise RuntimeError(f"GMO tokenisation failed (code {res.get('code') if isinstance(res, dict) else '?'})")
        return str(res["token"])

    def _interpret_checkout(self, raw: Any, biz: str, link: str) -> Optional[BookingResult]:
        """checkout / checkout_status body → result, or None while pending."""
        if not isinstance(raw, dict):
            raise SchemaDrift("Checkout response is not a JSON object.")
        res = raw.get("reservation")
        if isinstance(res, dict):
            code = res.get("slug") or res.get("confirmation_code") or res.get("id")
            return BookingResult(success=True, confirmation=str(code) if code else None,
                                 message=f"{biz} is booked, sir"
                                         f"{' — confirmation ' + str(code) if code else ''}.")
        st = raw.get("checkout_status")
        if isinstance(st, dict):
            state = str(st.get("status") or "")
            url = st.get("redirect_url") or st.get("payment_page_url")
            if state == "redirect_3ds" or url:
                return BookingResult(
                    success=False, needs_manual=True, error="3ds_pending",
                    message=(f"The card needs a 3-D Secure check to finish {biz}, sir. "
                             f"The table is held for about 10 minutes — complete it here: "
                             f"{url or link}"))
            if state == "pending":
                return None
            if state:
                return BookingResult(
                    success=False, needs_manual=True, error=f"checkout_{state}",
                    message=f"TableCheck reported '{state}' while booking {biz}, sir "
                            f"({st.get('message') or 'no detail'}). Finish it here: {link}")
        err = raw.get("error")
        if err:
            text = err.get("message") if isinstance(err, dict) else str(err)
            code = (err.get("code") or err.get("type")) if isinstance(err, dict) else None
            return BookingResult(
                success=False, needs_manual=True, error=f"checkout_error:{code or 'unknown'}",
                message=f"TableCheck couldn't complete the booking at {biz}, sir: "
                        f"{text or code or 'unknown error'}. Finish it here: {link}")
        avail = raw.get("availability")
        if isinstance(avail, dict) and avail.get("code"):
            return BookingResult(success=False, error="slot_not_found",
                                 message=f"That table at {biz} was taken as I booked it, sir.")
        raise SchemaDrift(f"Unrecognised checkout response keys {sorted(raw.keys())!r}.")

    async def _await_checkout(self, cart_id: str, biz: str, link: str,
                              first: Any) -> BookingResult:
        deadline = asyncio.get_event_loop().time() + float(
            os.getenv("TABLECHECK_CHECKOUT_WAIT_SECONDS", "90"))
        raw = first
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(3)
            status, raw = await self._call("PUT", f"/booking/checkout_status/{cart_id}", {})
            if status >= 500 or status == 429:
                continue
            result = self._interpret_checkout(raw, biz, link)
            if result is not None:
                return result
        return BookingResult(
            success=False, needs_manual=True, error="checkout_timeout",
            message=f"TableCheck is still processing the {biz} booking, sir — check "
                    f"here in a moment: {link}")

    async def _release_cart(self, cart_id: str) -> None:
        try:
            await self._call("DELETE", f"/booking/cart/{cart_id}")
        except Exception:
            logger.debug("Cart release failed (it expires on its own)", exc_info=True)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _slug_from_slots(slots: Dict[str, Any]) -> Optional[str]:
        decision = slots.get("channel_decision") or {}
        return (slug_from_url(decision.get("url"))
                or slug_from_url(slots.get("target_url")))

    @staticmethod
    def _flatten_errors(errors: Any) -> str:
        parts: List[str] = []

        def walk(node, prefix=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{prefix}{k}: " if not isinstance(v, (dict, list)) else prefix)
            elif isinstance(node, list):
                for v in node:
                    walk(v, prefix)
            else:
                parts.append(f"{prefix}{node}")
        walk(errors)
        return "; ".join(parts)[:400] or "unspecified"

    @staticmethod
    def _to_float(v: Any) -> float:
        try:
            return float(v or 0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _year4(year: Any) -> str:
        y = str(year).strip()
        return ("20" + y) if len(y) == 2 else y

    @staticmethod
    def _brand(pan: str) -> str:
        p = re.sub(r"\D", "", str(pan))
        if p.startswith("4"):
            return "visa"
        if p[:2] in {"34", "37"}:
            return "amex"
        if p.startswith("35"):
            return "jcb"
        if p[:2] in {"36", "38", "39"} or p[:3] in {"300", "301", "302", "303", "304", "305"}:
            return "diners"
        return "mastercard"
