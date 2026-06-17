"""OpenTable channel — Playwright + the user's logged-in session.

Inherits the BrowserChannel lifecycle/confirmation contract (and, with it, the
persistent-profile + session-cookie replay that `workflows.reservations.
browser_login opentable` captures — no extra storage is needed here). Unlike
Resy, OpenTable doesn't book inside an iframe widget: it navigates page-to-page
(restaurant profile → booking/details page → confirmation). So the flow drives
the public venue page: preselect date/party/time via query params, click the
exact time slot, then complete the reservation on the details page that loads.
Requires a logged-in OpenTable profile; an expired session degrades to a manual
hand-off, never a fake success. On any unexpected page state a screenshot + DOM
dump lands in the debug dir so failures are diagnosable after the window closes.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse

from ..models import ReservationMethod
from .base import BookingResult, BrowserChannel, CommitPlan

logger = logging.getLogger(__name__)

DEBUG_DIR = os.path.expanduser(
    os.getenv("RESERVATION_BROWSER_DEBUG_DIR", "~/.friday/browser-debug"))

# "7:00 PM" / "7:00PM" inside a slot button's label.
_TIME_12H = re.compile(r"(\d{1,2}):(\d{2})\s*(AM|PM)", re.I)
# Signals the booking went through. OpenTable's confirmation page reads
# "You're confirmed", "Reservation confirmed", "Your table is booked", etc.
_CONFIRMED_RE = re.compile(
    r"(you'?re (all set|confirmed)|reservation (is )?confirmed|reservation booked|"
    r"your table is (booked|confirmed)|booking (is )?confirmed|confirmed!|"
    r"we'?ll see you|thanks for your reservation)", re.I)
# A sign-in wall standing between us and the booking → session expired. The
# header always carries a "Sign in" link, so this is only trusted alongside a
# /signin URL or the absence of a Complete button (see _complete_booking).
_LOGIN_RE = re.compile(r"(sign ?in to (complete|continue|book)|log ?in to (complete|continue|book)|"
                       r"create (an )?account)", re.I)
# The card was rejected at "Complete" → an unresolvable card problem (the user
# needs a card with enough limit / a valid card), reported as a hand-off.
_CARD_DECLINE_RE = re.compile(
    r"(card (was )?declined|declined|could not (be )?process|unable to process|"
    r"problem with your (payment|card)|payment (failed|could not|error)|"
    r"invalid card|try (a )?(different|another) card|card (was )?not accepted)", re.I)


class OpenTableChannel(BrowserChannel):
    def __init__(self, profile_dir: str):
        super().__init__(ReservationMethod.OPENTABLE, "opentable", profile_dir)

    # ------------------------------------------------------------------ commit
    async def _do_booking(self, plan: CommitPlan, payment: Any = None) -> BookingResult:
        from playwright.async_api import async_playwright

        d = plan.details
        url = self._booking_url(d)
        if not url:
            return BookingResult(
                success=False, needs_manual=True,
                message="I don't have an OpenTable page for that one, sir.",
                error="no_url")

        async with async_playwright() as p:
            ctx = await self._launch(p)
            page = await ctx.new_page()
            try:
                await page.goto(url, timeout=45000)
                slot = await self._find_slot(page, d.get("time"))
                if slot is None:
                    await self._dump(page, "no_slot")
                    return BookingResult(
                        success=False, needs_manual=True,
                        message=(f"I couldn't find a {self._display_time(d.get('time'))} slot on "
                                 f"OpenTable for {d.get('business_name')}, sir — best check it "
                                 f"yourself: {url}."),
                        error="slot_not_found")

                # Clicking a slot navigates toward the booking flow. Many venues
                # insert a seating-options step before the details page.
                await slot.click()
                await self._select_seating(page)
                return await self._complete_booking(page, d, url, payment)
            finally:
                await ctx.close()

    async def _select_seating(self, page) -> None:
        """Handle the optional seating-options step (Standard / Bar / …).

        After the slot click some venues route to /booking/seating-options with
        a row of "Select" buttons; others go straight to /booking/details. Pick
        the default option (matching the slot's seating type) to proceed.

        OpenTable keeps analytics sockets open, so 'networkidle' never fires —
        wait on the URL/elements instead."""
        try:
            await page.wait_for_url("**/booking/**", timeout=15000)
        except Exception:
            pass
        if "/seating-options" not in (page.url or "").lower():
            return
        seat = page.locator("[data-test='seatingOption-default-button'], "
                            "[data-test^='seatingOption-'][data-test$='-button']")
        try:
            await seat.first.wait_for(state="visible", timeout=15000)
            await seat.first.click()
            await page.wait_for_url("**/booking/details**", timeout=15000)
        except Exception:
            logger.warning("OpenTable seating-option selection failed", exc_info=True)

    async def _complete_booking(self, page, d: Dict[str, Any], url: str,
                                payment: Any = None) -> BookingResult:
        """On the booking/details page: fill contact + (if required) card →
        Complete reservation → confirmation."""
        biz = d.get("business_name")

        # Settle on the details page (networkidle never fires — analytics sockets
        # stay open), then wait for the Complete button so the React form has
        # rendered before we touch the card fields.
        try:
            await page.wait_for_url("**/booking/details**", timeout=15000)
        except Exception:
            pass

        if self._on_signin(page):
            await self._dump(page, "login_required")
            return BookingResult(
                success=False, needs_manual=True,
                message=(f"OpenTable wants a fresh login before it'll book {biz}, sir — "
                         f"run `python -m workflows.reservations.browser_login opentable`, "
                         f"sign in once, and ask me again."),
                error="login_required")

        complete = page.locator(
            "[data-test='complete-reservation-button'], "
            "[data-testid='complete-reservation-button'], "
            "button[id*='complete-reservation'], #complete-reservation, "
            "button:has-text('Complete reservation'), "
            "button:has-text('Confirm reservation'), "
            "button:has-text('Reserve now'), "
            "button:has-text('Book now')")
        try:
            await complete.first.wait_for(state="visible", timeout=20000)
        except Exception:
            await self._dump(page, "no_complete_button")
            body = (await page.inner_text("body"))[:8000]
            if _LOGIN_RE.search(body):
                return BookingResult(
                    success=False, needs_manual=True,
                    message=(f"OpenTable wants a fresh login before it'll book {biz}, sir — "
                             f"run the login bootstrap and I'll try again."),
                    error="login_required")
            return BookingResult(
                success=False, needs_manual=True,
                message=(f"I opened the OpenTable booking page for {biz} but found no "
                         f"Complete-reservation button, sir. Finish here: {url}."),
                error="no_complete_button")

        body = (await page.inner_text("body"))[:8000]
        # Some flows land straight on a confirmation (e.g. an instant-book venue).
        if _CONFIRMED_RE.search(body):
            return self._booked(biz, body)

        # Top up any contact fields the logged-in session didn't prefill.
        await self._fill_contact(page, d)

        # Card-on-file / no-show guarantee: fill the card the gate minted upstream.
        if await self._card_required(page):
            if payment is None:
                await self._dump(page, "card_required")
                return BookingResult(
                    success=False, needs_manual=True,
                    message=(f"{biz} needs a card to hold the table, sir, but I don't "
                             f"have one to use. You'll need to finish this: {url}."),
                    error="card_required")
            if not await self._fill_card(page, payment, d):
                await self._dump(page, "card_fill_failed")
                return BookingResult(
                    success=False, needs_manual=True,
                    message=(f"I couldn't enter the card on OpenTable for {biz}, sir — "
                             f"please finish it here: {url}."),
                    error="card_fill_failed")

        await complete.first.click()

        # Wait for the confirmation page/state.
        deadline = time.time() + 30
        while time.time() < deadline:
            await page.wait_for_timeout(1500)
            if self._on_signin(page):
                # Complete bounced to sign-in → session expired mid-flow.
                await self._dump(page, "login_required")
                return BookingResult(
                    success=False, needs_manual=True,
                    message=(f"OpenTable wants a fresh login before it'll book {biz}, sir — "
                             f"run `python -m workflows.reservations.browser_login opentable`, "
                             f"sign in once, and ask me again."),
                    error="login_required")
            if "confirmation" in (page.url or "").lower():
                return self._booked(biz, None)
            try:
                body = (await page.inner_text("body"))[:8000]
            except Exception:
                continue
            if _CONFIRMED_RE.search(body):
                return self._booked(biz, body)
            if _CARD_DECLINE_RE.search(body):
                # The card the table needs was rejected — a card problem only the
                # user can resolve (a valid card with enough limit).
                await self._dump(page, "card_declined")
                return BookingResult(
                    success=False, needs_manual=True,
                    message=(f"OpenTable rejected the card for {biz}, sir — it needs a "
                             f"valid card to hold the table (no-show fee). Please book "
                             f"with your own card: {url}."),
                    error="card_declined")

        await self._dump(page, "unconfirmed")
        return BookingResult(
            success=False, needs_manual=True,
            message=(f"I clicked through OpenTable for {biz} but couldn't see a confirmation, "
                     f"sir — please verify before retrying: {url}."),
            error="unconfirmed")

    def _booked(self, biz: Optional[str], body: Optional[str]) -> BookingResult:
        return BookingResult(
            success=True,
            message=f"Booked, sir — {biz} is confirmed on OpenTable.",
            confirmation=self._extract_confirmation(body) if body else None)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _booking_url(d: Dict[str, Any]) -> Optional[str]:
        url = d.get("url")
        if not url or "opentable." not in urlparse(url).netloc:
            return None
        base = url.split("?")[0]
        params: Dict[str, Any] = {}
        date, hhmm = d.get("date"), d.get("time")
        if date and hhmm:
            # OpenTable's profile widget preselects from a combined ISO datetime.
            params["dateTime"] = f"{date}T{hhmm}"
        elif date:
            params["dateTime"] = date
        if d.get("party_size"):
            params["covers"] = d["party_size"]
        return f"{base}?{urlencode(params)}" if params else base

    async def _find_slot(self, page, hhmm: Optional[str]):
        """The slot button whose label shows exactly the requested time.

        OpenTable's profile widget renders indexed slot buttons
        (data-test="time-slot-0", "time-slot-1", …) inside a "time-slots"
        container; each carries the time in its aria-label, e.g.
        "Select table type for reservation at <venue> at 6:30 PM on …".
        """
        want = self._to_12h(hhmm)
        if want is None:
            return None
        buttons = page.locator("[data-test^='time-slot-'], [data-testid^='time-slot-']")
        try:
            # The grid is rendered twice (a responsive variant), so the *first*
            # element may be a hidden duplicate — wait for attachment, not
            # visibility, then prefer a visible match below.
            await buttons.first.wait_for(state="attached", timeout=25000)
        except Exception:
            return None
        fallback = None
        for i in range(await buttons.count()):
            b = buttons.nth(i)
            # The time lives in the aria-label; inner text may be just the time
            # or a table-type word — check the aria-label first, then the text.
            label = (await b.get_attribute("aria-label")) or ""
            if not _TIME_12H.search(label):
                label = (await b.inner_text() or "") or label
            m = _TIME_12H.search(label.replace(" ", " "))
            if m and (int(m.group(1)), m.group(2), m.group(3).upper()) == want:
                try:
                    if await b.is_visible():
                        return b
                except Exception:
                    pass
                fallback = fallback or b   # matched but hidden; use if nothing visible
        return fallback

    async def _fill_contact(self, page, d: Dict[str, Any]) -> None:
        """Best-effort top-up of contact fields the session didn't prefill.

        Logged-in OpenTable usually prefills name/phone from the account; this
        only fills a field that's present *and* empty, so it never clobbers the
        account's own details. Wrapped — it must never fail the booking path.
        """
        fields = {
            "phone": d.get("phone"),
            "firstName": d.get("guest_name"),
            "lastName": d.get("guest_name"),
            "email": d.get("email"),
        }
        for name, value in fields.items():
            if not value:
                continue
            try:
                loc = page.locator(f"input[name='{name}'], input[id*='{name}'], "
                                   f"input[autocomplete*='{name.lower()}']")
                if await loc.count() == 0:
                    continue
                field = loc.first
                if (await field.input_value()).strip():
                    continue  # account already filled it; leave it be
                await field.fill(str(value))
            except Exception:
                continue

    @staticmethod
    async def _card_required(page) -> bool:
        """Does this details page demand a card? OpenTable tokenizes the card in
        Spreedly iframes; their presence is the reliable signal. The details page
        renders async, so wait briefly for the field to attach."""
        loc = page.locator("iframe[id^='spreedly-number-frame-'], "
                           "input#creditCardName, input#creditCardZipCode")
        try:
            await loc.first.wait_for(state="attached", timeout=8000)
            return True
        except Exception:
            return False

    async def _fill_card(self, page, payment: Any, d: Dict[str, Any]) -> bool:
        """Enter the single-use card into OpenTable's Spreedly fields.

        Card number + CVV live in cross-origin Spreedly iframes; name, expiry and
        billing ZIP are plain inputs on the page; the venue's terms checkbox must
        be accepted. The PAN/CVV go only into the page DOM here — never logged or
        persisted (the card object guards its own repr)."""
        try:
            num = page.frame_locator("iframe[id^='spreedly-number-frame-']").locator("input")
            await num.first.wait_for(state="visible", timeout=15000)
            await num.first.fill(str(payment.pan))

            cvv = page.frame_locator("iframe[id^='spreedly-cvv-frame-']").locator("input")
            if await cvv.count() > 0:
                await cvv.first.fill(str(payment.cvv))

            name = d.get("guest_name") or os.getenv("RESERVATION_GUEST_NAME") or "Cardholder"
            await self._fill_if_present(page, "input#creditCardName", str(name))

            mm = str(payment.exp_month).zfill(2)
            yy = str(payment.exp_year)[-2:]
            await self._fill_if_present(page, "input#creditCardExpiry", f"{mm} / {yy}")

            # Billing ZIP — some venues (Copra) require it. Privacy single-use
            # cards ignore AVS, so the configured ZIP just needs to be present.
            zip_code = getattr(payment, "zip_code", "") or os.getenv("RESERVATION_CARD_ZIP", "")
            if zip_code:
                await self._fill_if_present(page, "input#creditCardZipCode", str(zip_code))

            # Accept the restaurant's terms (required to complete).
            await self._accept_terms(page)
            return True
        except Exception:
            logger.warning("OpenTable card entry failed", exc_info=True)
            return False

    @staticmethod
    async def _fill_if_present(page, selector: str, value: str) -> None:
        loc = page.locator(selector)
        if await loc.count() > 0:
            await loc.first.fill(value)

    @staticmethod
    async def _accept_terms(page) -> None:
        """Tick the venue terms checkbox (custom-styled, so click the label if
        the input itself isn't directly checkable)."""
        box = page.locator("#tcAccepted")
        if await box.count() == 0:
            return
        try:
            await box.first.check(force=True)
            if await box.first.is_checked():
                return
        except Exception:
            pass
        for sel in ("label[for='tcAccepted']", "#tcAccepted + label",
                    "#tcAccepted ~ label"):
            try:
                lbl = page.locator(sel)
                if await lbl.count() > 0:
                    await lbl.first.click()
                    if await box.first.is_checked():
                        return
            except Exception:
                continue

    @staticmethod
    def _on_signin(page) -> bool:
        path = urlparse(page.url or "").path.lower()
        return "/signin" in path or "/login" in path

    @staticmethod
    def _to_12h(hhmm: Optional[str]):
        """'19:00' → (7, '00', 'PM'), matching slot-button labels."""
        if not hhmm:
            return None
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", hhmm.strip())
        if not m:
            return None
        h, mins = int(m.group(1)), m.group(2)
        return ((h - 12) or 12, mins, "PM") if h >= 12 else (h or 12, mins, "AM")

    @staticmethod
    def _display_time(hhmm: Optional[str]) -> str:
        t = OpenTableChannel._to_12h(hhmm)
        return f"{t[0]}:{t[1]} {t[2]}" if t else (hhmm or "the requested time")

    @staticmethod
    def _extract_confirmation(body: str) -> Optional[str]:
        m = re.search(r"(?i)confirmation\s*(?:#|number|code)?[:\s]*([A-Z0-9\-]{4,})", body)
        return m.group(1) if m else None

    async def _dump(self, page, tag: str) -> None:
        """Screenshot + DOM snapshot for post-mortem; never fails the booking path."""
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            await page.screenshot(path=os.path.join(DEBUG_DIR, f"opentable-{stamp}-{tag}.png"),
                                  full_page=True)
            with open(os.path.join(DEBUG_DIR, f"opentable-{stamp}-{tag}.html"), "w") as fh:
                fh.write(await page.content())
            logger.warning("OpenTable debug dump written: %s (%s)", DEBUG_DIR, tag)
        except Exception:
            logger.warning("OpenTable debug dump failed", exc_info=True)
