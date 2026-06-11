"""
ReservationLLM — small structured-completion helper for the reservations agent.

Three uses (all optional refinements over deterministic fallbacks):
  - slot extraction from the opening utterance (fallback: regexes in workflow.py),
  - discovery method classification (fallback: heuristic in discovery.py),
  - email drafting (fallback: template in channels/email.py).

`from_env()` returns None when ANTHROPIC_API_KEY or the SDK is missing, and every
helper returns None/falls back on any failure, so nothing here is load-bearing.
Card data is never included in any prompt (spec §8 L4), and classification
prompts carry only business evidence — no user PII (L5).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


class ReservationLLM:
    """One-shot, history-free JSON completions (the personality-laden
    LLMProvider in llm/providers.py is unsuitable for structured extraction)."""

    def __init__(self, client, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model

    @classmethod
    def from_env(cls) -> Optional["ReservationLLM"]:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            import anthropic
        except Exception:
            logger.warning("anthropic SDK unavailable; reservation LLM refinement off")
            return None
        model = os.getenv("RESERVATION_LLM_MODEL", DEFAULT_MODEL)
        return cls(anthropic.Anthropic(api_key=api_key), model)

    def complete_json(self, system: str, user: str,
                      max_tokens: int = 700) -> Optional[Dict[str, Any]]:
        """Returns the parsed JSON object, or None on any failure."""
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )
            return _parse_json(text)
        except Exception:
            logger.warning("Reservation LLM call failed", exc_info=True)
            return None


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Tolerates code fences and prose around the JSON object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


# --------------------------------------------------------------------- slot extraction

_EXTRACT_SYSTEM = """You extract reservation details from a request to a voice assistant.
Reply with ONLY a JSON object — no prose. Keys (use null when not stated; never guess):
  business_name: the establishment's name (NOT generic words like "a table" or "a reservation")
  date: the requested date, verbatim as said (e.g. "next Friday", "tomorrow", "6/14")
  time: the requested time, verbatim (e.g. "7pm", "half past seven")
  party_size: integer number of people
  service_type: e.g. "dinner", "haircut", "60-minute massage" (null if not stated)
  special_requests: any notes (window seat, allergies, stylist preference...)
  location: city/neighbourhood if the user states one"""

_STR_SLOTS = ("business_name", "date", "time", "service_type", "special_requests", "location")


def extract_slots(llm: Optional[ReservationLLM], text: str) -> Optional[Dict[str, Any]]:
    """LLM slot extraction. Returns a validated dict, or None (→ caller's regex fallback)."""
    if llm is None:
        return None
    raw = llm.complete_json(_EXTRACT_SYSTEM, text, max_tokens=300)
    if raw is None:
        return None

    slots: Dict[str, Any] = {}
    for key in _STR_SLOTS:
        v = raw.get(key)
        if isinstance(v, str):
            v = v.strip()
            if v and len(v) <= 120 and v.lower() not in ("null", "none", "n/a"):
                slots[key] = v
    party = raw.get("party_size")
    if isinstance(party, str) and party.isdigit():
        party = int(party)
    if isinstance(party, int) and 1 <= party <= 100:
        slots["party_size"] = party

    # A business name made of filler words means the model ignored the rules.
    biz = slots.get("business_name", "").lower()
    if biz and any(w in biz for w in ("reservation", "booking", "appointment", "table")):
        slots.pop("business_name")
    return slots or None


# ----------------------------------------------------------------- method classification

_CLASSIFY_SYSTEM = """You determine HOW a business accepts reservations, from lookup evidence.
Reply with ONLY a JSON object:
  method: one of "opentable" | "resy" | "yelp" | "generic_web" | "phone" | "email" | "unknown"
  url: the booking URL (must be copied from the evidence) or null
  email: the booking email address (must appear in the evidence) or null
  requires_card: true if the evidence mentions a deposit / card hold / prepayment
  confidence: 0.0–1.0
  notes: one short sentence of reasoning
Rules:
- "opentable"/"resy" only when the booking itself is hosted on that platform.
- Appointment platforms (Vagaro, Booksy, Square, Mindbody, Fresha, Tock, SevenRooms)
  and a business's own booking form are "generic_web".
- "email" only when email is the stated way to request a booking (e.g. "email us to reserve").
- "phone" when there is no online booking but a phone number exists.
- Never invent URLs or email addresses not present in the evidence."""

_VALID_METHODS = {"opentable", "resy", "yelp", "generic_web", "phone", "email", "unknown"}


def classify_method(llm: Optional[ReservationLLM],
                    evidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """LLM method classification over discovery evidence. Returns a validated dict
    {method, url, email, requires_card, confidence, notes} or None (→ heuristic)."""
    if llm is None:
        return None
    evidence_json = json.dumps(evidence, ensure_ascii=False, default=str)
    raw = llm.complete_json(_CLASSIFY_SYSTEM, evidence_json, max_tokens=300)
    if raw is None or raw.get("method") not in _VALID_METHODS:
        return None

    out: Dict[str, Any] = {"method": raw["method"]}

    # Anti-hallucination: contact details must literally appear in the evidence.
    url = raw.get("url")
    out["url"] = url if isinstance(url, str) and url in evidence_json else None
    email = raw.get("email")
    if isinstance(email, str) and _EMAIL_RE.fullmatch(email) and email in evidence_json:
        out["email"] = email
    else:
        out["email"] = None
    if out["method"] == "email" and not out["email"]:
        return None  # an email decision without an address is useless

    out["requires_card"] = bool(raw.get("requires_card"))
    try:
        out["confidence"] = min(1.0, max(0.0, float(raw.get("confidence", 0.5))))
    except (TypeError, ValueError):
        out["confidence"] = 0.5
    notes = raw.get("notes")
    out["notes"] = notes.strip()[:300] if isinstance(notes, str) else ""
    return out


# ------------------------------------------------------------------------ email drafting

_DRAFT_SYSTEM = """You draft a short, polite reservation-request email on behalf of a guest.
Reply with ONLY a JSON object: {"subject": "...", "body": "..."}.
Plain text body, no markdown. Include the party size, date, time (and flexibility if
given), the guest's name, and their callback phone if provided. Ask them to confirm
availability or suggest the nearest alternative. Never mention payment card details."""


def make_llm_drafter(llm: ReservationLLM, fallback) -> "callable":
    """Wraps an LLM into the EmailChannel drafter signature, falling back to the
    template drafter on any failure."""

    def drafter(slots: Dict[str, Any], decision,
                instruction: Optional[str] = None) -> Tuple[str, str]:
        facts = {
            "business": getattr(decision, "business_name", None) or slots.get("business_name"),
            "date": slots.get("date"), "time": slots.get("time"),
            "party_size": slots.get("party_size"),
            "guest_name": slots.get("guest_name"),
            "callback_phone": slots.get("phone"),
            "service_type": slots.get("service_type"),
            "special_requests": slots.get("special_requests"),
        }
        if instruction:
            facts["revision_instruction_from_guest"] = instruction
        raw = llm.complete_json(_DRAFT_SYSTEM, json.dumps(facts, default=str), max_tokens=600)
        if raw and isinstance(raw.get("subject"), str) and isinstance(raw.get("body"), str):
            return raw["subject"].strip(), raw["body"].strip()
        return fallback(slots, decision, instruction=instruction)

    return drafter
