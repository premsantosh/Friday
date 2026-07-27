"""
FormAgent — harvest, map, fill, submit, verify.

The split between planning and execution is what makes this safe to put behind
the confirmation gate:

  `plan()`  loads the page and works out exactly what would be typed where. It
            is a GET on a public page and touches nothing. Its output is shown
            to the user field by field.
  `execute()` re-loads the page, checks the form hasn't changed underneath the
            approved plan, fills, submits, and then tries hard to find out
            whether it actually worked.

Verification is deliberately pessimistic. A single weak signal is reported as
`unconfirmed` with the URL for the user to check, never as success — the same
rule the OpenTable channel follows. Nothing here ever claims a booking it
cannot evidence.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.profile import DEFAULT_CONTEXT, UserProfile, choose_context, match_key

from .harvest import BLOCK_CARD, BLOCK_PASSWORD, FormField, FormSnapshot, harvest, locator_for
from .mapper import Assignment, FieldMap, MissingFact, match_option, map_fields

logger = logging.getLogger(__name__)

MAX_STEPS = int(os.getenv("RESERVATION_FORM_MAX_STEPS", "3"))

_SUBMIT_RE = re.compile(
    r"(?i)\b(book|reserve|confirm|request|submit|send|continue|next|complete|schedule|"
    r"get started|make (?:an )?appointment)\b")
_SUCCESS_TEXT_RE = re.compile(
    r"(?i)(thank you|thanks for (?:your|reaching)|we(?:'| ha)ve received|message (?:has been )?sent|"
    r"request (?:has been )?(?:received|submitted|sent)|successfully (?:sent|submitted|booked)|"
    r"appointment (?:request )?(?:received|confirmed|booked)|reservation confirmed|"
    r"we(?:'| wi)ll be in touch|someone will (?:contact|reach out)|you'?re all set)")
_SUCCESS_URL_RE = re.compile(
    r"(?i)(thank|success|confirm|submitted|complete|received|done)")
_ERROR_TEXT_RE = re.compile(
    r"(?i)(this field is required|is required\b|please (?:enter|select|provide|fill|complete)|"
    r"required field|invalid |not a valid|must be a valid|please correct)")
_CONFIRMATION_RE = re.compile(
    r"(?i)(?:confirmation|reference|booking|request)\s*(?:#|no\.?|number|code|id)?[:\s#]*"
    r"([A-Z0-9][A-Z0-9\-]{3,})")


# --------------------------------------------------------------------- plan types

@dataclass
class PlanEntry:
    """One field we will fill, with the exact value that will be typed."""
    ref: str
    label: str
    field_type: str
    value: str
    origin: str                       # "profile:<key>" | "request" | "choice" | "consent"
    option_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"ref": self.ref, "label": self.label, "field_type": self.field_type,
                "value": self.value, "origin": self.origin, "option_value": self.option_value}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlanEntry":
        return cls(ref=d["ref"], label=d.get("label", ""),
                   field_type=d.get("field_type", "text"), value=d.get("value", ""),
                   origin=d.get("origin", "request"), option_value=d.get("option_value"))


@dataclass
class FormPlan:
    url: str = ""
    heading: str = ""
    entries: List[PlanEntry] = field(default_factory=list)
    missing: List[MissingFact] = field(default_factory=list)
    # Drift detection: the shape of the form at the moment the user approved it.
    known_refs: List[str] = field(default_factory=list)
    required_refs: List[str] = field(default_factory=list)
    submit_labels: List[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    # Which persona filled it — shown at the gate so a wrong one is obvious
    # before submission, not after.
    context: str = DEFAULT_CONTEXT
    # A bot-detection widget gates submission; we can fill, but the click has
    # to be the user's and even then the site may score the session as a bot.
    human_submit_required: Optional[str] = None
    provenance: str = "none"
    notes: str = ""

    @property
    def ok(self) -> bool:
        return self.blocked_reason is None and bool(self.entries) and not self.missing

    def describe(self) -> str:
        """The field-by-field disclosure shown at the confirmation gate."""
        if not self.entries:
            return "  (nothing to fill)"
        width = min(34, max((len(e.label or e.ref) for e in self.entries), default=10))
        lines = []
        for e in self.entries:
            label = (e.label or e.ref)[:width].ljust(width)
            lines.append(f"  {label}  {e.value}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"url": self.url, "heading": self.heading,
                "entries": [e.to_dict() for e in self.entries],
                "missing": [m.to_dict() for m in self.missing],
                "known_refs": self.known_refs, "required_refs": self.required_refs,
                "submit_labels": self.submit_labels, "blocked_reason": self.blocked_reason,
                "context": self.context,
                "human_submit_required": self.human_submit_required,
                "provenance": self.provenance, "notes": self.notes}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FormPlan":
        return cls(url=d.get("url", ""), heading=d.get("heading", ""),
                   entries=[PlanEntry.from_dict(e) for e in d.get("entries") or []],
                   missing=[MissingFact.from_dict(m) for m in d.get("missing") or []],
                   known_refs=list(d.get("known_refs") or []),
                   required_refs=list(d.get("required_refs") or []),
                   submit_labels=list(d.get("submit_labels") or []),
                   blocked_reason=d.get("blocked_reason"),
                   context=d.get("context", DEFAULT_CONTEXT),
                   human_submit_required=d.get("human_submit_required"),
                   provenance=d.get("provenance", "none"), notes=d.get("notes", ""))


@dataclass
class FormOutcome:
    status: str          # submitted | unconfirmed | validation_failed | drift | blocked
                         # | no_submit | nothing_to_fill | missing_facts | filled_only
    message: str = ""
    filled: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    confirmation: Optional[str] = None
    url: str = ""

    @property
    def success(self) -> bool:
        return self.status == "submitted"


_BLOCK_MESSAGES = {
    BLOCK_PASSWORD: "that form asks for a password",
    BLOCK_CARD: "that form asks for card details",
}


class FormAgent:
    """Drives an arbitrary web form from the durable profile + booking request."""

    def __init__(self, llm=None, profile: Optional[UserProfile] = None,
                 max_steps: int = MAX_STEPS):
        self.llm = llm
        self._profile = profile
        self.max_steps = max(1, max_steps)

    @property
    def profile(self) -> UserProfile:
        if self._profile is None:
            self._profile = UserProfile()
            self._profile.seed_from_env()
        return self._profile

    # ------------------------------------------------------------------- plan
    async def plan(self, page, request: Dict[str, Any]) -> FormPlan:
        """Read the form on the current page and decide what to put in it."""
        snapshot = await harvest(page)
        if snapshot.blocked_reason:
            return FormPlan(url=snapshot.url, heading=snapshot.heading,
                            blocked_reason=snapshot.blocked_reason)
        if not snapshot.fields:
            return FormPlan(url=snapshot.url, heading=snapshot.heading,
                            notes="no fillable fields found")

        context = self._context_for(snapshot, request)
        profile = self.profile
        field_map = map_fields(self.llm, snapshot, profile.descriptors(context),
                               _request_for_llm(request))
        return self._build_plan(snapshot, field_map, profile, context)

    @staticmethod
    def _context_for(snapshot: FormSnapshot, request: Dict[str, Any]) -> str:
        """Which persona fills this form.

        The form's own fields are the strongest evidence — a page asking for a
        date of birth or an insurance carrier keeps records under your legal
        name. Failing that, the business and service description decide.
        """
        field_keys = {match_key(f"{f.label} {f.placeholder} {f.name}")
                      for f in snapshot.fields}
        text = " ".join(str(v) for v in (
            snapshot.heading, snapshot.url,
            (request or {}).get("business_name"), (request or {}).get("service_type"),
        ) if v)
        return choose_context(text, field_keys)

    def _build_plan(self, snapshot: FormSnapshot, field_map: FieldMap,
                    profile: UserProfile, context: str) -> FormPlan:
        """Substitute real values locally. This is where facts stop being keys."""
        wanted = [a.fact_key for a in field_map.assignments
                  if a.source == "fact" and a.fact_key]
        values = profile.values(wanted, context)

        entries: List[PlanEntry] = []
        missing = list(field_map.missing)
        for assignment in field_map.assignments:
            fld = snapshot.by_ref(assignment.ref)
            if fld is None:
                continue

            if assignment.source == "fact" and not values.get(assignment.fact_key or ""):
                # We promised a fact we turn out not to hold — worth asking for.
                missing.append(MissingFact(
                    ref=fld.ref, label=fld.label, suggested_key=assignment.fact_key,
                    options=[o.label for o in fld.options if o.value]))
                continue

            entry = self._entry_for(assignment, fld, values)
            if entry is None:
                # We hold a value, but it doesn't fit this field — typically a
                # dropdown with a fixed menu the stored answer isn't on. Asking
                # again would just yield the same non-matching answer, so an
                # optional field is left blank and a required one hands off.
                if fld.required:
                    missing.append(MissingFact(ref=fld.ref, label=fld.label,
                                               suggested_key=None,
                                               options=[o.label for o in fld.options if o.value]))
                continue
            entries.append(entry)

        return FormPlan(
            url=snapshot.url, heading=snapshot.heading, entries=entries, missing=missing,
            known_refs=[f.ref for f in snapshot.fields],
            required_refs=snapshot.required_refs(),
            submit_labels=snapshot.submit_labels, context=context,
            human_submit_required=snapshot.human_submit_required,
            provenance=field_map.provenance, notes=field_map.notes,
        )

    @staticmethod
    def _entry_for(assignment: Assignment, fld: FormField,
                   values: Dict[str, str]) -> Optional[PlanEntry]:
        if assignment.source == "fact":
            value = values.get(assignment.fact_key or "")
            if not value:
                return None
            origin = f"profile:{assignment.fact_key}"
        else:
            value = assignment.literal or ""
            if not value:
                return None
            origin = ("choice" if fld.describes_choice
                      else "consent" if fld.type == "checkbox" else "request")

        option_value = assignment.option_value
        if fld.describes_choice and option_value is None:
            # A fact filling a dropdown still has to land on a real option, and
            # the plan must show the option the user will actually get.
            option_value = match_option(fld, value)
            if option_value is None:
                return None
            value = next((o.label for o in fld.options if o.value == option_value), value)

        return PlanEntry(ref=fld.ref, label=fld.label or fld.ref, field_type=fld.type,
                         value=value, origin=origin, option_value=option_value)

    # ---------------------------------------------------------------- execute
    async def execute(self, page, plan: FormPlan, *, submit: bool = True) -> FormOutcome:
        """Replay an approved plan against a freshly loaded page."""
        if plan.blocked_reason:
            return FormOutcome(status="blocked", url=plan.url,
                               message=_BLOCK_MESSAGES.get(plan.blocked_reason,
                                                           "that form can't be filled safely"))
        if not plan.entries:
            return FormOutcome(status="nothing_to_fill", url=plan.url)

        current = await harvest(page)
        drift = _drift(plan, current)
        if drift:
            return FormOutcome(status="drift", url=plan.url, message=drift)

        filled, skipped, errors = await self._apply(page, plan, current)
        if not filled:
            return FormOutcome(status="drift", url=plan.url, skipped=skipped, errors=errors,
                               message="none of the approved fields could be filled")
        if not submit or plan.human_submit_required:
            # Submitting here would be rejected: these widgets score the whole
            # session, and a programmatically-filled form fails regardless of
            # who clicks. Fill it and hand the click back rather than burn an
            # attempt and report a failure we could have predicted.
            return FormOutcome(status="filled_only", url=page.url, filled=filled,
                               skipped=skipped, errors=errors,
                               message=plan.human_submit_required or "")

        before = _PageState(url=page.url, refs=set(current.required_refs()))
        clicked = await self._submit(page, current)
        if not clicked:
            return FormOutcome(status="no_submit", url=page.url, filled=filled,
                               skipped=skipped, errors=errors,
                               message="couldn't find a submit button")

        outcome = await self._verify(page, before)
        outcome.filled, outcome.skipped = filled, skipped
        outcome.errors.extend(errors)
        return outcome

    async def _apply(self, page, plan: FormPlan, current: FormSnapshot):
        filled: List[str] = []
        skipped: List[str] = []
        errors: List[str] = []
        for entry in plan.entries:
            fld = current.by_ref(entry.ref)
            if fld is None:
                skipped.append(entry.ref)
                continue
            try:
                if await self._fill_one(page, fld, entry):
                    filled.append(entry.ref)
                else:
                    skipped.append(entry.ref)
            except Exception as exc:
                logger.warning("Couldn't fill %s", entry.ref, exc_info=True)
                skipped.append(entry.ref)
                errors.append(f"{entry.ref}: {exc}")
        return filled, skipped, errors

    @staticmethod
    async def _fill_one(page, fld: FormField, entry: PlanEntry) -> bool:
        if fld.type == "radio":
            locator = await locator_for(page, fld, option_value=entry.option_value)
            if locator is None:
                return False
            await locator.check()
            return True

        locator = await locator_for(page, fld)
        if locator is None:
            return False

        if fld.type == "select":
            value = entry.option_value or entry.value
            try:
                await locator.select_option(value=value)
            except Exception:
                await locator.select_option(label=entry.value)
            return True

        if fld.type == "checkbox":
            if entry.value.strip().lower() == "true":
                await locator.check()
            else:
                await locator.uncheck()
            return True

        await locator.fill(entry.value)
        return True

    async def _submit(self, page, snapshot: FormSnapshot) -> bool:
        """Click the button that submits this form. No blind Enter presses."""
        candidates = [label for label in snapshot.submit_labels if _SUBMIT_RE.search(label)]
        for label in candidates:
            try:
                button = page.get_by_role("button", name=label, exact=True)
                if await button.count() == 0:
                    button = page.get_by_text(label, exact=True)
                for i in range(min(await button.count(), 5)):
                    target = button.nth(i)
                    if await target.is_visible():
                        await target.click()
                        await page.wait_for_timeout(3500)
                        return True
            except Exception:
                continue
        for selector in ('button[type="submit"]', 'input[type="submit"]'):
            try:
                locator = page.locator(selector)
                for i in range(min(await locator.count(), 5)):
                    target = locator.nth(i)
                    if await target.is_visible():
                        await target.click()
                        await page.wait_for_timeout(3500)
                        return True
            except Exception:
                continue
        return False

    async def _verify(self, page, before: "_PageState") -> FormOutcome:
        """Did it actually go through? Two independent signals, or we don't claim it."""
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=8000)
        except Exception:
            pass

        signals: List[str] = []
        url = page.url
        try:
            body = (await page.inner_text("body"))[:8000]
        except Exception:
            body = ""

        changed = url != before.url
        if changed and _SUCCESS_URL_RE.search(url):
            signals.append("success_url")
        if _SUCCESS_TEXT_RE.search(body):
            signals.append("success_text")

        after = await harvest(page)
        remaining = set(after.required_refs())
        errors_shown = bool(_ERROR_TEXT_RE.search(body)) or await _has_invalid_field(page)
        if before.refs and not (before.refs & remaining):
            signals.append("form_gone")

        if errors_shown and (before.refs & remaining):
            return FormOutcome(
                status="validation_failed", url=url, signals=signals,
                message=_first_error(body) or "the form reported a validation error")

        confirmation = _confirmation_from(body) if signals else None
        if len(signals) >= 2:
            return FormOutcome(status="submitted", url=url, signals=signals,
                               confirmation=confirmation)

        # Ambiguous: keep the evidence so "did that go through?" is answerable
        # later, rather than only a shrug in the log.
        await dump_debug(page, "unconfirmed")
        return FormOutcome(status="unconfirmed", url=url, signals=signals,
                           message="submitted but couldn't confirm it went through")

    # ----------------------------------------------------------------- one-shot
    async def run(self, page, request: Dict[str, Any], *, submit: bool = True) -> tuple:
        """Plan and execute in one pass, walking multi-step forms.

        Returns (plan, outcome). Used by the dry-run CLI and by any caller that
        doesn't need to gate between planning and execution — the channel does
        gate, so it calls plan() and execute() separately.
        """
        plan = await self.plan(page, request)
        if not plan.entries or plan.blocked_reason or plan.missing:
            status = ("blocked" if plan.blocked_reason
                      else "missing_facts" if plan.missing else "nothing_to_fill")
            return plan, FormOutcome(status=status, url=plan.url)

        outcome = await self.execute(page, plan, submit=submit)
        step = 1
        # A "next"-style page: the submit advanced us to another form rather
        # than a confirmation. Re-plan against whatever is on screen now.
        while (submit and step < self.max_steps and outcome.status == "unconfirmed"
               and not outcome.signals):
            next_plan = await self.plan(page, request)
            if not next_plan.entries or next_plan.missing:
                break
            next_outcome = await self.execute(page, next_plan, submit=True)
            if next_outcome.status in ("drift", "nothing_to_fill"):
                break
            plan, outcome = next_plan, next_outcome
            step += 1
        return plan, outcome


# ------------------------------------------------------------------- helpers

@dataclass
class _PageState:
    url: str
    refs: set


def _drift(plan: FormPlan, current: FormSnapshot) -> Optional[str]:
    """Refuse to replay a plan against a form that changed shape underneath it."""
    if current.blocked_reason:
        return _BLOCK_MESSAGES.get(current.blocked_reason, "the form can't be filled safely")
    if not current.fields:
        return "the form is no longer on that page"
    present = {f.ref for f in current.fields}
    gone = [e.ref for e in plan.entries if e.ref not in present]
    if gone:
        return f"the form changed — {', '.join(gone[:3])} is no longer there"
    new_required = set(current.required_refs()) - set(plan.known_refs)
    if new_required:
        return (f"the form now asks for something new: "
                f"{', '.join(sorted(new_required)[:3])}")
    return None


def _request_for_llm(request: Dict[str, Any]) -> Dict[str, Any]:
    """Booking facts minus the guest's identity — that arrives as fact keys."""
    drop = {"guest_name", "phone", "email", "raw_request"}
    return {k: v for k, v in (request or {}).items() if k not in drop}


async def _watch_for_submission(page, agent: FormAgent, before: "_PageState",
                                hold_seconds: int) -> FormOutcome:
    """Wait for a human to press submit, then report what actually happened.

    The one step Friday won't take is the click itself. Everything after it —
    did it go through, did the form reject it — is still verified the same way
    an automated submission would be, so the user gets a real answer rather
    than "I filled it in, good luck".
    """
    deadline = hold_seconds * 1000
    waited = 0
    while waited < deadline:
        await page.wait_for_timeout(2000)
        waited += 2000
        try:
            body = (await page.inner_text("body"))[:8000]
        except Exception:
            continue                       # mid-navigation
        moved = page.url != before.url
        if moved or _SUCCESS_TEXT_RE.search(body) or _ERROR_TEXT_RE.search(body):
            return await agent._verify(page, before)
        try:
            still_there = {f.ref for f in (await harvest(page)).fields}
        except Exception:
            continue
        if before.refs and not (before.refs & still_there):
            return await agent._verify(page, before)
    return FormOutcome(status="no_submit", url=page.url,
                       message="no submission detected")


async def _has_invalid_field(page) -> bool:
    try:
        return await page.locator('[aria-invalid="true"]').count() > 0
    except Exception:
        return False


def _first_error(body: str) -> Optional[str]:
    match = _ERROR_TEXT_RE.search(body or "")
    if not match:
        return None
    start = max(0, match.start() - 60)
    return " ".join((body[start:match.end() + 60]).split())


def _confirmation_from(body: str) -> Optional[str]:
    match = _CONFIRMATION_RE.search(body or "")
    return match.group(1) if match else None


async def dump_debug(page, tag: str) -> Optional[str]:
    """Screenshot + HTML when something unexpected happens, as the other
    channels do. Never raises."""
    directory = os.path.expanduser(
        os.getenv("RESERVATION_BROWSER_DEBUG_DIR", "~/.friday/browser-debug"))
    try:
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = os.path.join(directory, f"form-{tag}-{stamp}")
        await page.screenshot(path=f"{base}.png", full_page=True)
        with open(f"{base}.html", "w") as fh:
            fh.write(await page.content())
        return base
    except Exception:
        logger.debug("Couldn't write form debug dump", exc_info=True)
        return None


# ------------------------------------------------------------------------ CLI

async def _dry_run(url: str, request: Dict[str, Any], do_fill: bool,
                   hold: int = 300) -> int:
    from playwright.async_api import async_playwright

    from ..channels.base import persistent_context_options
    from ..llm import ReservationLLM

    profile_dir = os.path.join(
        os.path.expanduser(os.getenv("RESERVATION_BROWSER_DIR", "~/.friday/browser")),
        "generic_web")
    os.makedirs(profile_dir, exist_ok=True)
    agent = FormAgent(llm=ReservationLLM.from_env())

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            **persistent_context_options(profile_dir))
        page = await ctx.new_page()
        try:
            await page.goto(url, timeout=45000)
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass                       # analytics sockets often never idle

            plan = await agent.plan(page, request)

            print(f"\n  page      {plan.url}")
            print(f"  heading   {plan.heading}")
            print(f"  mapped by {plan.provenance}"
                  + (f" — {plan.notes}" if plan.notes else ""))
            if plan.blocked_reason:
                print(f"\n  REFUSED: {_BLOCK_MESSAGES.get(plan.blocked_reason)}\n")
                return 2

            print(f"\n  would fill {len(plan.entries)} field(s):\n")
            for entry in plan.entries:
                origin = f"[{entry.origin}]"
                print(f"    {(entry.label or entry.ref)[:38]:38} {origin:24} {entry.value}")

            if plan.missing:
                print(f"\n  missing {len(plan.missing)} fact(s) — would ask you for these:\n")
                for miss in plan.missing:
                    hint = f"  (profile key: {miss.suggested_key})" if miss.suggested_key else ""
                    print(f"    {(miss.label or miss.ref)[:38]:38}{hint}")

            print(f"\n  submit buttons seen: {', '.join(plan.submit_labels) or '(none)'}")

            if do_fill and plan.entries:
                snapshot = await harvest(page)
                filled, skipped, errors = await agent._apply(page, plan, snapshot)
                print(f"\n  filled {len(filled)}, skipped {len(skipped)}"
                      + (f", errors: {errors}" if errors else ""))
                button = plan.submit_labels[0] if plan.submit_labels else "the submit button"
                print(f"\n  Everything is filled in. Click “{button}” in the browser "
                      f"window to send it.")
                print("  Friday will not click it: this form uses reCAPTCHA, which exists "
                      "to require\n  a human, and defeating that isn't something I'll do.")
                print(f"  Watching for {hold}s to confirm whether it went through...\n")

                before = _PageState(url=page.url, refs=set(snapshot.required_refs()))
                outcome = await _watch_for_submission(page, agent, before, hold)
                if outcome.status == "submitted":
                    print(f"  ✅ Submitted and confirmed ({', '.join(outcome.signals)})."
                          + (f" Reference: {outcome.confirmation}" if outcome.confirmation else ""))
                    return 0
                if outcome.status == "validation_failed":
                    print(f"  ⚠️  Their form rejected it: {outcome.message}")
                    return 3
                print("  Nothing was submitted — no click detected before the timeout.\n")
                return 4

            print("\n  Nothing was submitted.\n")
            return 0
        finally:
            await ctx.close()


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        prog="python -m workflows.reservations.formfill",
        description="Read a booking/contact form and show what Friday would put in it.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="plan only, never submit (the default, and the only mode)")
    parser.add_argument("--fill", action="store_true",
                        help="type the values into the page and wait for you to press submit")
    parser.add_argument("--hold", type=int, default=300,
                        help="seconds to keep the window open with --fill (default: %(default)s)")
    parser.add_argument("--service", default="", help="what the appointment is for")
    parser.add_argument("--date", default="", help="preferred date (ISO)")
    parser.add_argument("--time", default="", help="preferred time (HH:MM)")
    parser.add_argument("--notes", default="", help="anything else to say")
    parser.add_argument("--business", default="", help="business name, for context")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        pass

    logging.basicConfig(level=logging.WARNING)
    request = {"business_name": args.business, "service_type": args.service,
               "date": args.date, "time": args.time, "special_requests": args.notes}
    return asyncio.run(_dry_run(args.url, request, args.fill, args.hold))
