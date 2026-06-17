"""
GenericWebChannel — our own Playwright form engine (preferred no-API path).

Best-effort: opens the booking URL, fills fields by matching common
labels/placeholders/names to reservation slots, clicks a book/reserve/confirm
button, and looks for a success signal. Simple forms may go through; complex
SPAs gracefully hand off (the base wraps exceptions into a manual hand-off).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..models import ReservationMethod
from .base import BookingResult, BrowserChannel, CommitPlan

logger = logging.getLogger(__name__)

# Reservation slot -> candidate field identifiers (matched against name/id/placeholder/label).
_FIELD_HINTS = {
    "guest_name": ["name", "full name", "your name", "fname"],
    "email": ["email", "e-mail"],
    "phone": ["phone", "mobile", "tel"],
    "party_size": ["party", "guests", "people", "size", "covers"],
    "date": ["date", "day"],
    "time": ["time"],
    "special_requests": ["notes", "requests", "special", "comments"],
}
_SUBMIT_RE = r"(?i)\b(book|reserve|confirm|request|submit|continue|complete)\b"
_SUCCESS_RE = r"(?i)(confirmed|reservation .*(confirmed|received)|booking .*(confirmed|received)|you'?re all set|thank you)"


class GenericWebChannel(BrowserChannel):
    def __init__(self, profile_dir: str):
        super().__init__(ReservationMethod.GENERIC_WEB, "generic_web", profile_dir)

    async def _do_booking(self, plan: CommitPlan, payment: Any = None) -> BookingResult:
        import re

        from playwright.async_api import async_playwright

        url = plan.details.get("url")
        if not url:
            return BookingResult(success=False, needs_manual=True,
                                 message="I don't have a booking URL for that one, sir.",
                                 error="no_url")

        async with async_playwright() as p:
            ctx = await self._launch(p)
            page = await ctx.new_page()
            try:
                await page.goto(url, timeout=30000)
                await self._autofill(page, plan.details)

                button = page.get_by_role("button", name=re.compile(_SUBMIT_RE))
                if await button.count() == 0:
                    return BookingResult(
                        success=False, needs_manual=True,
                        message=(f"I filled what I could on {plan.details.get('business_name')} but "
                                 f"couldn't find a booking button, sir. Finish here: {url}."),
                        error="no_submit",
                    )
                await button.first.click()
                await page.wait_for_timeout(3000)

                body = (await page.inner_text("body"))[:5000]
                if re.search(_SUCCESS_RE, body):
                    return BookingResult(
                        success=True,
                        message=f"Booked, sir — {plan.details.get('business_name')} is confirmed.",
                        confirmation=self._extract_confirmation(body),
                    )
                return BookingResult(
                    success=False, needs_manual=True,
                    message=(f"I submitted the form for {plan.details.get('business_name')} but "
                             f"couldn't confirm it went through, sir. Please verify: {url}."),
                    error="unconfirmed",
                )
            finally:
                await ctx.close()

    async def _autofill(self, page, details: Dict[str, Any]) -> None:
        import re
        for slot, hints in _FIELD_HINTS.items():
            value = details.get(slot)
            if value in (None, ""):
                continue
            pattern = re.compile("|".join(re.escape(h) for h in hints), re.I)
            for getter in (page.get_by_label, page.get_by_placeholder):
                try:
                    field = getter(pattern)
                    if await field.count() > 0:
                        await field.first.fill(str(value))
                        break
                except Exception:
                    continue

    @staticmethod
    def _extract_confirmation(body: str):
        import re
        m = re.search(r"(?i)confirmation\s*(?:#|number|code)?[:\s]*([A-Z0-9\-]{4,})", body)
        return m.group(1) if m else None
