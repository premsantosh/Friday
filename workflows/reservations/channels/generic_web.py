"""
GenericWebChannel — any business's own booking, appointment or contact form.

This is the no-API path, and now the broadest one: a physical therapy clinic's
consultation request, a salon's Vagaro widget, a restaurant's own reservation
page. It carries no per-site selectors. `formfill/` reads whatever form is on
the page, decides which of the user's durable facts answers each field, and
fills it.

The gate contract the workflow depends on:

  `prepare()` loads the page and returns a plan that names **every field and the
  exact value** that would be typed. That plan is what the user approves, and
  `hash_plan` binds the approval to it.

  `commit()` reloads, checks the form still matches the approved plan, fills,
  submits, and verifies. A form that changed shape is handed back to the user
  rather than improvised at.

A submitted contact form is a *request*, not a confirmed booking — the outcome
says so (`pending=True`) instead of claiming a reservation the business hasn't
agreed to yet.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from core.harness import display_date, display_time

from ..formfill import FormAgent, FormPlan
from ..models import ChannelDecision, ReservationMethod
from .base import BookingResult, BrowserChannel, CommitPlan

logger = logging.getLogger(__name__)

# Forms that only *request* a slot rather than confirm one. Used to phrase the
# outcome honestly: a restaurant's own instant-booking page confirms, a clinic's
# "request a consultation" does not.
_REQUEST_WORDS = ("request", "inquiry", "enquiry", "contact", "consultation",
                  "get started", "reach out", "message")


class GenericWebChannel(BrowserChannel):
    def __init__(self, profile_dir: str, agent: Optional[FormAgent] = None,
                 llm: Any = None):
        super().__init__(ReservationMethod.GENERIC_WEB, "generic_web", profile_dir)
        self.agent = agent or FormAgent(llm=llm)

    # ------------------------------------------------------------------ prepare
    async def prepare(self, slots: Dict[str, Any], decision: ChannelDecision) -> CommitPlan:
        """Read the form and work out what would go in it. Submits nothing."""
        plan = await super().prepare(slots, decision)
        plan.details["request"] = _request_from(slots, decision)
        url = plan.details.get("url")
        if not url:
            plan.details["form_error"] = "no_url"
            return plan

        try:
            form_plan = await self._plan_form(url, plan.details["request"])
        except Exception as exc:
            # A page that won't load is not fatal here: commit() re-plans, and
            # the user still gets a link. Never let it kill the dialogue.
            logger.warning("Couldn't read the form at %s", url, exc_info=True)
            plan.details["form_error"] = str(exc)
            return plan

        plan.details["form_plan"] = form_plan.to_dict()
        if form_plan.blocked_reason:
            plan.details["form_error"] = form_plan.blocked_reason
        plan.summary = self._summary(slots, decision, form_plan)
        return plan

    async def _plan_form(self, url: str, request: Dict[str, Any]) -> FormPlan:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            ctx = await self._launch(p)
            page = await ctx.new_page()
            try:
                await page.goto(url, timeout=45000)
                await settle(page)
                return await self.agent.plan(page, request)
            finally:
                await ctx.close()

    @staticmethod
    def _summary(slots: Dict[str, Any], decision: ChannelDecision,
                 form_plan: FormPlan) -> str:
        """The one-liner above the field list at the confirmation gate."""
        business = decision.business_name or slots.get("business_name")
        what = slots.get("service_type") or "an appointment"
        verb = "Submit a request for" if is_request_form(form_plan) else "Book"
        # Resolved, human dates: the user approves "Friday, July 31", not the
        # ISO string or the ambiguous phrase they originally said.
        when = ""
        if slots.get("date"):
            when = f" on {display_date(slots['date'])}"
            if slots.get("time"):
                when += f" at {display_time(slots['time'])}"
        return f"{verb} {what} at {business}{when} via their own booking form"

    # ------------------------------------------------------------------- commit
    async def _do_booking(self, plan: CommitPlan, payment: Any = None) -> BookingResult:
        url = plan.details.get("url")
        business = plan.details.get("business_name") or "them"
        if not url:
            return BookingResult(success=False, needs_manual=True, error="no_url",
                                 message="I don't have a booking URL for that one, sir.")

        stored = plan.details.get("form_plan")
        approved = FormPlan.from_dict(stored) if stored else None

        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            ctx = await self._launch(p)
            page = await ctx.new_page()
            try:
                await page.goto(url, timeout=45000)
                await settle(page)

                if approved is None or not approved.entries:
                    # prepare() couldn't read the form. Re-plan now rather than
                    # submit something nobody has seen.
                    approved = await self.agent.plan(page, plan.details.get("request") or {})
                    if not approved.entries:
                        return self._handoff(business, url, approved.blocked_reason
                                             or "I couldn't read their form")

                if approved.blocked_reason:
                    return self._handoff(business, url, approved.blocked_reason)
                if approved.missing:
                    wanted = ", ".join(m.label or m.ref for m in approved.missing[:3])
                    return self._handoff(business, url,
                                         f"the form needs details I don't have ({wanted})")

                outcome = await self.agent.execute(page, approved, submit=True)
                return self._result(outcome, business, url, approved)
            finally:
                await ctx.close()

    def _result(self, outcome, business: str, url: str, form_plan: FormPlan) -> BookingResult:
        if outcome.status == "submitted":
            if is_request_form(form_plan):
                return BookingResult(
                    success=True, pending=True, confirmation=outcome.confirmation,
                    message=(f"Request submitted to {business}, sir — they have your "
                             f"details and will be in touch to confirm a time."))
            return BookingResult(
                success=True, confirmation=outcome.confirmation,
                message=f"Booked, sir — {business} is confirmed.")

        if outcome.status == "validation_failed":
            return self._handoff(business, url,
                                 f"their form rejected it ({outcome.message})",
                                 error="validation_failed")

        if outcome.status == "unconfirmed":
            return BookingResult(
                success=False, needs_manual=True, error="unconfirmed",
                message=(f"I filled and submitted {business}'s form, sir, but couldn't "
                         f"confirm it went through. Best verify here: {url}."))

        if outcome.status == "drift":
            return self._handoff(business, url, outcome.message, error="form_changed")

        if outcome.status == "filled_only":
            guard = outcome.message or "a bot check"
            return BookingResult(
                success=False, needs_manual=True, error="human_submit_required",
                message=(
                    f"I've filled in {business}'s form for you, sir, but their "
                    f"{guard} check requires a person to send it — and it scores the "
                    f"whole session, so it rejects the form even when you click it "
                    f"yourself after I've typed it. You'll need to fill this one in "
                    f"by hand: {url}. I have every value ready if you'd like them."))

        if outcome.status == "no_submit":
            return self._handoff(business, url, "I couldn't find a submit button",
                                 error="no_submit")

        return self._handoff(business, url, outcome.message or "I couldn't complete it",
                             error=outcome.status)

    @staticmethod
    def _handoff(business: str, url: str, reason: str,
                 error: str = "needs_manual") -> BookingResult:
        return BookingResult(
            success=False, needs_manual=True, error=error,
            message=(f"I stopped short of submitting {business}'s form, sir — {reason}. "
                     f"You can finish it here: {url}."))


# ---------------------------------------------------------------------- helpers

def is_request_form(form_plan: Optional[FormPlan]) -> bool:
    """Does this form request an appointment, or actually book one?"""
    if form_plan is None:
        return True                      # the safer assumption
    text = f"{form_plan.heading} {' '.join(form_plan.submit_labels)} {form_plan.url}".lower()
    return any(word in text for word in _REQUEST_WORDS)


def _request_from(slots: Dict[str, Any], decision: ChannelDecision) -> Dict[str, Any]:
    """The booking facts the mapper may see. Identity is deliberately absent —
    name, phone and email reach the form as profile fact *keys*, not values."""
    return {
        "business_name": decision.business_name or slots.get("business_name"),
        "service_type": slots.get("service_type"),
        "date": slots.get("date"),
        "time": slots.get("time"),
        "party_size": slots.get("party_size"),
        "special_requests": slots.get("special_requests"),
    }


async def settle(page) -> None:
    """Give a JS-rendered form a chance to appear.

    `networkidle` frequently never fires on marketing sites (analytics sockets),
    so this is a best-effort wait with a short ceiling — the same lesson the
    OpenTable channel encodes.
    """
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=15000)
    except Exception:
        pass
    try:
        await page.wait_for_selector("form, input, textarea", timeout=8000)
    except Exception:
        pass
    await page.wait_for_timeout(1200)
