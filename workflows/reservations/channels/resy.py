"""Resy channel — Playwright + the user's logged-in session.

Shares the BrowserChannel lifecycle/confirmation contract. The booking flow
drives the public venue page: preselect date/party via query params, click the
exact time slot, then complete the "Reserve Now" widget that Resy serves in an
iframe. Requires a logged-in Resy profile (workflows.reservations.browser_login
resy); an expired session degrades to a manual hand-off, never a fake success.
On any unexpected page state a screenshot + DOM dump lands in the debug dir so
failures are diagnosable after the window closes.
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

# "6:00 PM" / "6:00PM" inside a slot button's label.
_TIME_12H = re.compile(r"(\d{1,2}):(\d{2})\s*(AM|PM)", re.I)
# Signals that the widget finished the booking. Resy's success panel reads
# "Reservation Booked." with "Please check your inbox for a confirmation email"
# and a "Continue to Reservation Details" CTA — none of which contain the word
# "confirmed", so match the booked/inbox/CTA wording too (not just "confirmed").
_CONFIRMED_RE = re.compile(
    r"(reservation (is )?confirmed|reservation booked|booking (is )?(confirmed|booked)|"
    r"you'?re all set|confirmed!|check your (inbox|email) for (a |your )?confirmation|"
    r"continue to reservation details)", re.I)
# The widget asking for an account instead of a summary → session expired.
_LOGIN_RE = re.compile(r"(log ?in|sign ?up|create (an )?account|continue with)", re.I)


class ResyChannel(BrowserChannel):
    def __init__(self, profile_dir: str):
        super().__init__(ReservationMethod.RESY, "resy", profile_dir)

    # ------------------------------------------------------------------ commit
    async def _do_booking(self, plan: CommitPlan, payment: Any = None) -> BookingResult:
        from playwright.async_api import async_playwright

        d = plan.details
        url = self._booking_url(d)
        if not url:
            return BookingResult(
                success=False, needs_manual=True,
                message="I don't have a Resy page for that one, sir.",
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
                                 f"Resy for {d.get('business_name')}, sir — best check it "
                                 f"yourself: {url}."),
                        error="slot_not_found")

                await slot.click()
                frame = await self._widget_frame(page)
                if frame is None:
                    await self._dump(page, "no_widget")
                    return BookingResult(
                        success=False, needs_manual=True,
                        message=(f"Resy didn't open its booking panel for "
                                 f"{d.get('business_name')}, sir. Finish here: {url}."),
                        error="widget_not_found")

                return await self._complete_widget(page, frame, d, url)
            finally:
                await ctx.close()

    async def _complete_widget(self, page, frame, d: Dict[str, Any], url: str) -> BookingResult:
        """Inside the widgets.resy.com iframe: Reserve Now → (optional) Confirm → done."""
        biz = d.get("business_name")

        body = await self._frame_text(frame)
        if body and _LOGIN_RE.search(body) and "Reserve" not in body:
            await self._dump(page, "login_required")
            return BookingResult(
                success=False, needs_manual=True,
                message=(f"Resy wants a fresh login before it'll book {biz}, sir — "
                         f"run the login bootstrap and I'll try again."),
                error="login_required")

        reserve = frame.locator(
            "[data-test-id='order_summary_page-button-book'], "
            "button:has-text('Reserve Now'), button:has-text('Book Now')")
        try:
            await reserve.first.wait_for(state="visible", timeout=20000)
        except Exception:
            await self._dump(page, "no_reserve_button")
            return BookingResult(
                success=False, needs_manual=True,
                message=(f"I opened the Resy booking panel for {biz} but found no "
                         f"Reserve button, sir. Finish here: {url}."),
                error="no_reserve_button")
        await reserve.first.click()

        # Some venues add a second step (cancellation policy / "Confirm").
        deadline = time.time() + 30
        while time.time() < deadline:
            await page.wait_for_timeout(1500)
            body = await self._frame_text(frame)
            if body is None:                       # widget closed itself
                break
            if _CONFIRMED_RE.search(body):
                return BookingResult(
                    success=True,
                    message=f"Booked, sir — {biz} is confirmed on Resy.",
                    confirmation=None)
            if _LOGIN_RE.search(body) and not _TIME_12H.search(body):
                # Reserve Now bounced to phone-verification → session expired.
                await self._dump(page, "login_required")
                return BookingResult(
                    success=False, needs_manual=True,
                    message=(f"Resy wants a fresh login before it'll book {biz}, sir — "
                             f"run `python -m workflows.reservations.browser_login resy`, "
                             f"sign in once, and ask me again."),
                    error="login_required")
            confirm = frame.locator(
                "button:has-text('Confirm'):not([disabled]), "
                "[data-test-id='order_summary_page-button-confirm']")
            try:
                if await confirm.count() > 0 and await confirm.first.is_visible():
                    await confirm.first.click()
                    continue
            except Exception:
                pass

        # The widget may close on success and the venue page show the booking.
        page_text = (await page.inner_text("body"))[:8000]
        if _CONFIRMED_RE.search(page_text):
            return BookingResult(
                success=True,
                message=f"Booked, sir — {biz} is confirmed on Resy.",
                confirmation=None)

        await self._dump(page, "unconfirmed")
        return BookingResult(
            success=False, needs_manual=True,
            message=(f"I clicked through Resy for {biz} but couldn't see a confirmation, "
                     f"sir — please verify before retrying: {url}."),
            error="unconfirmed")

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _booking_url(d: Dict[str, Any]) -> Optional[str]:
        url = d.get("url")
        if not url or "resy.com" not in urlparse(url).netloc:
            return None
        base = url.split("?")[0]
        params = {}
        if d.get("date"):
            params["date"] = d["date"]
        if d.get("party_size"):
            params["seats"] = d["party_size"]
        return f"{base}?{urlencode(params)}" if params else base

    async def _find_slot(self, page, hhmm: Optional[str]):
        """The slot button whose label shows exactly the requested time."""
        want = self._to_12h(hhmm)
        if want is None:
            return None
        buttons = page.locator(
            "[data-test-id='reservation-button'], button.ReservationButton, "
            "a.ReservationButton")
        try:
            await buttons.first.wait_for(state="visible", timeout=25000)
        except Exception:
            return None
        for i in range(await buttons.count()):
            b = buttons.nth(i)
            m = _TIME_12H.search((await b.inner_text()).replace(" ", " "))
            if m and (int(m.group(1)), m.group(2), m.group(3).upper()) == want:
                return b
        return None

    @staticmethod
    def _to_12h(hhmm: Optional[str]):
        """'18:00' → (6, '00', 'PM'), matching slot-button labels."""
        if not hhmm:
            return None
        m = re.fullmatch(r"(\d{1,2}):(\d{2})", hhmm.strip())
        if not m:
            return None
        h, mins = int(m.group(1)), m.group(2)
        return ((h - 12) or 12, mins, "PM") if h >= 12 else (h or 12, mins, "AM")

    @staticmethod
    def _display_time(hhmm: Optional[str]) -> str:
        t = ResyChannel._to_12h(hhmm)
        return f"{t[0]}:{t[1]} {t[2]}" if t else (hhmm or "the requested time")

    @staticmethod
    async def _widget_frame(page):
        """Resy's booking flow runs in a widgets.resy.com iframe; find it."""
        for _ in range(20):
            for f in page.frames:
                if "widgets.resy.com" in (f.url or ""):
                    return f
            await page.wait_for_timeout(1000)
        return None

    @staticmethod
    async def _frame_text(frame) -> Optional[str]:
        try:
            return (await frame.inner_text("body"))[:8000]
        except Exception:
            return None

    async def _dump(self, page, tag: str) -> None:
        """Screenshot + DOM snapshot for post-mortem; never fails the booking path."""
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            await page.screenshot(path=os.path.join(DEBUG_DIR, f"resy-{stamp}-{tag}.png"),
                                  full_page=True)
            with open(os.path.join(DEBUG_DIR, f"resy-{stamp}-{tag}.html"), "w") as fh:
                fh.write(await page.content())
            logger.warning("Resy debug dump written: %s (%s)", DEBUG_DIR, tag)
        except Exception:
            logger.warning("Resy debug dump failed", exc_info=True)
