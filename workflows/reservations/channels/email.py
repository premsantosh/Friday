"""
Email channel — reservation-by-email (M5b).

Entered when a business is email-only (`method: email`) or when a phone call
returns `email_requested`. Flow: draft → user approves/edits → send (from the
user's own mailbox) → session WAITING → on_tick polls *only the reservation
thread* for a reply → classify (confirmed / declined / needs-info / asks-to-call).

Drafter, sender, and reader are injectable so the workflow is testable without a
mail server; real SMTP/IMAP are best-effort and no-op when unconfigured. The
agent never scans the broader inbox (see spec §8, L11).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

from ..models import ChannelDecision, ReservationMethod
from .base import Availability, AvailabilityStatus, BookingResult, CommitPlan, ReservationChannel

logger = logging.getLogger(__name__)

EMAIL_POLL_SECONDS = 1800  # replies take a while; poll every 30 min


class EmailOutcome(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DECLINED = "declined"
    NEEDS_INFO = "needs_info"
    ASKS_TO_CALL = "asks_to_call"


@dataclass
class EmailReply:
    outcome: EmailOutcome
    text: str = ""
    confirmation: Optional[str] = None


def default_drafter(slots: Dict[str, Any], decision: ChannelDecision,
                    instruction: Optional[str] = None) -> Tuple[str, str]:
    """Template draft (subject, body). An LLM drafter can be injected instead."""
    biz = decision.business_name or slots.get("business_name")
    party = slots.get("party_size")
    date = slots.get("date")
    time = slots.get("time")
    name = slots.get("guest_name") or "Your guest"
    phone = slots.get("phone") or ""
    extras = slots.get("special_requests")

    subject = f"Reservation request — {name}, {date} {time}"
    lines = [
        f"Hello{f' {biz}' if biz else ''},",
        "",
        f"I'd like to request a reservation for {party} on {date} at {time}.",
    ]
    if extras:
        lines.append(f"Note: {extras}.")
    if instruction:
        lines.append(instruction)  # incorporate the user's edit
    lines += [
        "",
        "Could you confirm availability, or let me know the nearest time you have?",
        f"You can reach me at {phone}." if phone else "",
        "",
        f"Thank you,\n{name}",
    ]
    return subject, "\n".join(l for l in lines if l is not None)


class EmailChannel(ReservationChannel):
    method = ReservationMethod.EMAIL
    can_commit = True
    is_async = True
    is_email = True

    def __init__(self, sender=None, reader=None,
                 drafter: Optional[Callable] = None,
                 classifier: Optional[Callable[[str], EmailOutcome]] = None):
        self.sender = sender            # .send(to, subject, body) -> bool
        self.reader = reader            # .fetch_reply(to, subject) -> Optional[str]
        self.drafter = drafter or default_drafter
        self.classifier = classifier or self._classify_reply

    @classmethod
    def from_env(cls) -> Optional["EmailChannel"]:
        provider = os.getenv("RESERVATION_EMAIL_PROVIDER", "off").lower()
        if provider == "off":
            return None
        sender = _smtp_sender_from_env() if provider == "smtp" else None
        reader = _imap_reader_from_env()
        if sender is None:
            return None  # configured but unusable → not selectable
        return cls(sender=sender, reader=reader)

    # ----- ReservationChannel surface (commit unused; workflow drives the async flow)
    async def check_availability(self, slots: Dict[str, Any]) -> Availability:
        return Availability(status=AvailabilityStatus.UNKNOWN, note="I'll email a request.")

    async def prepare(self, slots: Dict[str, Any], decision: ChannelDecision) -> CommitPlan:
        to = decision.email or slots.get("business_email")
        # Off the event loop: the drafter may be LLM-backed.
        subject, body = await asyncio.to_thread(self.drafter, slots, decision)
        return CommitPlan(
            channel="email",
            summary=f"Email a reservation request to {decision.business_name}",
            details={
                "business_name": decision.business_name or slots.get("business_name"),
                "to": to, "subject": subject, "body": body,
                "date": slots.get("date"), "time": slots.get("time"),
                "party_size": slots.get("party_size"),
            },
            requires_card=False,
        )

    async def commit(self, plan: CommitPlan, payment: Any = None) -> BookingResult:
        return BookingResult(success=False, needs_manual=True,
                             message="Email is sent asynchronously.", error="use_async_flow")

    # ----- send / poll
    def send(self, plan: CommitPlan) -> bool:
        if self.sender is None:
            return False
        to = plan.details.get("to")
        if not to:
            return False
        try:
            return bool(self.sender.send(to, plan.details["subject"], plan.details["body"]))
        except Exception:
            logger.warning("Email send failed.", exc_info=True)
            return False

    def poll_reply(self, to: str, subject: str) -> EmailReply:
        if self.reader is None:
            return EmailReply(EmailOutcome.PENDING)
        try:
            text = self.reader.fetch_reply(to, subject)
        except Exception:
            logger.warning("Email reply fetch failed.", exc_info=True)
            return EmailReply(EmailOutcome.PENDING)
        if not text:
            return EmailReply(EmailOutcome.PENDING)
        return EmailReply(self.classifier(text), text=text,
                          confirmation=self._extract_confirmation(text))

    @staticmethod
    def _classify_reply(text: str) -> EmailOutcome:
        t = text.lower()

        def has(*w):
            return any(x in t for x in w)

        if has("call us", "give us a call", "please call", "phone us"):
            return EmailOutcome.ASKS_TO_CALL
        if has("unfortunately", "fully booked", "no availability", "unable to", "cannot accommodate",
               "we're full", "we are full"):
            return EmailOutcome.DECLINED
        if has("confirmed", "you're booked", "see you", "we have you", "reservation is set",
               "look forward to"):
            return EmailOutcome.CONFIRMED
        return EmailOutcome.NEEDS_INFO

    @staticmethod
    def _extract_confirmation(text: str) -> Optional[str]:
        m = re.search(r"(?i)confirmation\s*(?:#|number|code)?[:\s]*([A-Z0-9\-]{4,})", text)
        return m.group(1) if m else None


# --------------------------------------------------------------------------- real I/O

def _smtp_sender_from_env():
    host = os.getenv("SMTP_HOST")
    frm = os.getenv("RESERVATION_EMAIL_FROM")
    if not (host and frm):
        return None

    class _SmtpSender:
        def send(self, to: str, subject: str, body: str) -> bool:
            import smtplib
            from email.message import EmailMessage

            msg = EmailMessage()
            msg["From"] = frm
            msg["To"] = to
            msg["Subject"] = subject
            msg.set_content(body)
            port = int(os.getenv("SMTP_PORT", "587"))
            with smtplib.SMTP(host, port, timeout=20) as s:
                s.starttls()
                user, pw = os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD")
                if user and pw:
                    s.login(user, pw)
                s.send_message(msg)
            return True

    return _SmtpSender()


def _imap_reader_from_env():
    host = os.getenv("IMAP_HOST")
    user = os.getenv("IMAP_USER")
    pw = os.getenv("IMAP_PASSWORD")
    if not (host and user and pw):
        return None

    class _ImapReader:
        def fetch_reply(self, to: str, subject: str) -> Optional[str]:
            import email
            import imaplib

            with imaplib.IMAP4_SSL(host) as m:
                m.login(user, pw)
                m.select("INBOX")
                # Scope strictly to this reservation thread (subject), never the whole inbox.
                key = subject.replace('"', "")
                typ, data = m.search(None, 'SUBJECT', f'"{key}"')
                if typ != "OK" or not data or not data[0]:
                    return None
                latest = data[0].split()[-1]
                typ, msg_data = m.fetch(latest, "(RFC822)")
                if typ != "OK":
                    return None
                msg = email.message_from_bytes(msg_data[0][1])
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            return part.get_payload(decode=True).decode(errors="ignore")
                    return None
                return msg.get_payload(decode=True).decode(errors="ignore")

    return _ImapReader()
