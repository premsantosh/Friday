"""
Reservation LLM tasks — declarations only.

The generic machinery (JSON client, schema validation, grounding, fallback
provenance, egress checks) lives in core/harness/extract.py. This module
declares the three tasks the reservations agent uses, all optional refinements
over deterministic fallbacks:

  - slot extraction from the opening utterance (fallback: regexes in workflow.py),
  - discovery method classification (fallback: heuristic in discovery.py),
  - email drafting (fallback: template in channels/email.py).

Anti-hallucination is declarative: `url`/`email` are grounded fields — they
are dropped unless they literally appear in the evidence. Card data is never
included in any prompt (spec §8 L4); classification prompts carry only
business evidence — no user PII (L5). Both rules are enforced by the sinks.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from core.harness import (
    FieldSpec,
    JsonLLMClient,
    LLMTask,
    Sink,
    SinkMode,
    guard,
    run_task,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# Prompts are free text (the utterance/slots are the point), but nothing
# card- or secret-shaped may ever reach the provider (spec §8 L4); the
# classification sink additionally allows only business evidence (L5).
LLM_TEXT_SINK = Sink("llm", SinkMode.SCAN)
LLM_CLASSIFY_SINK = Sink("llm.classify", SinkMode.ALLOWLIST,
                         frozenset({"business", "search_results"}))


class ReservationLLM(JsonLLMClient):
    """One-shot, history-free JSON completions (the personality-laden
    LLMProvider in llm/providers.py is unsuitable for structured extraction)."""

    @classmethod
    def from_env(cls) -> Optional["ReservationLLM"]:
        base = JsonLLMClient.from_env(model_env="RESERVATION_LLM_MODEL",
                                      default_model=DEFAULT_MODEL)
        return cls(base.client, base.model) if base is not None else None


# --------------------------------------------------------------------- slot extraction

_EXTRACT_SYSTEM = """You extract booking details from a request to a voice assistant.
Reply with ONLY a JSON object — no prose. Keys (use null when not stated; never guess):
  business_name: the establishment's name (NOT generic words like "a table" or "a reservation")
  booking_kind: one of
    "dining"      — a table at a restaurant/bar (needs a party size and a time)
    "appointment" — a service booked for a time: clinic, therapist, salon, spa,
                    dentist, tattoo, repair, test drive, tour
    "inquiry"     — reaching out to a business with no specific time offered
  target_url: a URL the user gave for the business, copied exactly from their
    message (null if they gave none)
  date: the requested date, verbatim as said (e.g. "next Friday", "tomorrow", "6/14")
  time: the requested time, verbatim (e.g. "7pm", "half past seven")
  party_size: integer number of people (dining only; null for an appointment)
  service_type: what the booking is FOR, e.g. "dinner", "haircut",
    "60-minute massage", "physical therapy consultation" (null if not stated)
  special_requests: any notes (window seat, allergies, stylist preference...)
  location: city/neighbourhood if the user states one
  release_days_ahead: integer, ONLY if the user says how many days in advance
    bookings open (e.g. "tables drop 30 days in advance" -> 30); else null
  release_time: the time bookings open, verbatim, ONLY if stated
    (e.g. "at 10am", "9:00 ET"); else null
  release_timezone: a timezone the user attaches to the release time
    (e.g. "ET", "PT", "America/New_York"); else null"""


def _filler_business(name: str) -> bool:
    """A business name made of filler words means the model ignored the rules."""
    return any(w in name.lower() for w in ("reservation", "booking", "appointment", "table"))


def _coerce_party(v: Any) -> Optional[int]:
    if isinstance(v, str) and v.isdigit():
        v = int(v)
    return v if isinstance(v, int) and not isinstance(v, bool) else None


EXTRACT_TASK = LLMTask(
    name="slot_extraction",
    system_prompt=_EXTRACT_SYSTEM,
    sink=LLM_TEXT_SINK,
    max_tokens=300,
    fields=(
        FieldSpec("business_name", str, max_len=120, reject_if=_filler_business),
        FieldSpec("booking_kind", str, max_len=20,
                  valid=lambda k: k in ("dining", "appointment", "inquiry")),
        # Grounded: a URL we'd navigate to must have come from the user, not
        # from the model's imagination.
        FieldSpec("target_url", str, grounded=True, max_len=300,
                  valid=lambda u: u.startswith(("http://", "https://"))),
        FieldSpec("date", str, max_len=120),
        FieldSpec("time", str, max_len=120),
        FieldSpec("service_type", str, max_len=120),
        FieldSpec("special_requests", str, max_len=120),
        FieldSpec("location", str, max_len=120),
        FieldSpec("party_size", int, coerce=_coerce_party,
                  valid=lambda n: 1 <= n <= 100),
        FieldSpec("release_days_ahead", int, coerce=_coerce_party,
                  valid=lambda n: 1 <= n <= 365),
        FieldSpec("release_time", str, max_len=40),
        FieldSpec("release_timezone", str, max_len=40),
    ),
)


def extract_slots(llm: Optional[ReservationLLM], text: str) -> Optional[Dict[str, Any]]:
    """LLM slot extraction. Returns a validated dict, or None (→ caller's regex fallback).

    The utterance is also the evidence, so `target_url` is only accepted when
    the user actually said it."""
    result = run_task(llm, EXTRACT_TASK, text, evidence=text)
    return result.values if result.from_llm else None


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


def _coerce_confidence(v: Any) -> float:
    return min(1.0, max(0.0, float(v)))


def _finalize_classification(values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    values.setdefault("url", None)
    values.setdefault("email", None)
    values.setdefault("requires_card", False)
    values.setdefault("confidence", 0.5)
    values.setdefault("notes", "")
    if values["method"] == "email" and not values["email"]:
        return None  # an email decision without a grounded address is useless
    return values


CLASSIFY_TASK = LLMTask(
    name="method_classification",
    system_prompt=_CLASSIFY_SYSTEM,
    sink=LLM_TEXT_SINK,     # payload is the evidence JSON; allowlist applied upstream
    max_tokens=300,
    fields=(
        FieldSpec("method", str, required=True, valid=lambda m: m in _VALID_METHODS),
        FieldSpec("url", str, grounded=True),
        FieldSpec("email", str, grounded=True,
                  valid=lambda e: bool(_EMAIL_RE.fullmatch(e))),
        FieldSpec("requires_card", bool, coerce=bool),
        FieldSpec("confidence", float, coerce=_coerce_confidence),
        FieldSpec("notes", str, max_len=300),
    ),
    finalize=_finalize_classification,
)


def classify_method(llm: Optional[ReservationLLM],
                    evidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """LLM method classification over discovery evidence. Returns a validated dict
    {method, url, email, requires_card, confidence, notes} or None (→ heuristic).
    Contact details are grounded fields: hallucinated ones are dropped."""
    if llm is None:
        return None
    guard(LLM_CLASSIFY_SINK, evidence)
    evidence_json = json.dumps(evidence, ensure_ascii=False, default=str)
    result = run_task(llm, CLASSIFY_TASK, evidence_json, evidence=evidence_json)
    return result.values if result.from_llm else None


# ------------------------------------------------------- reservation-release policy

_RELEASE_SYSTEM = """You determine WHEN a venue releases its reservations, from
search evidence (which may include forum/Reddit threads). Reply with ONLY a JSON object:
  opens_days_in_advance: integer days before the dining date that booking opens
    (e.g. "tables drop 30 days out" -> 30) or null
  release_time: the local clock time bookings open, copied verbatim from the
    evidence (e.g. "10am", "9:00 AM") or null
  release_timezone: the timezone for that time, copied from the evidence
    (e.g. "ET", "PT", "America/New_York") or null
  rolling: true if a new day opens each day at that lead time; false for a batch
    drop (e.g. all of next month at once)
  confidence: 0.0-1.0 — lower it when sources disagree or are old/anecdotal
  source_quote: the sentence from the evidence that states the policy
  notes: one short sentence (mention if this came from a forum rather than the venue)
Rules:
- Use ONLY the evidence; never guess a time or day count that isn't stated.
- Prefer the venue's own/official statement; treat forum posts as corroboration
  and lower confidence when they're the only source or conflict with each other."""


def _finalize_release(values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # A usable policy needs both the lead days and the clock time to fire on.
    if values.get("opens_days_in_advance") is None or not values.get("release_time"):
        return None
    values.setdefault("release_timezone", None)
    values.setdefault("rolling", True)
    values.setdefault("confidence", 0.5)
    values.setdefault("source_quote", "")
    values.setdefault("notes", "")
    return values


RELEASE_POLICY_TASK = LLMTask(
    name="release_policy",
    system_prompt=_RELEASE_SYSTEM,
    sink=LLM_TEXT_SINK,
    max_tokens=400,
    fields=(
        FieldSpec("opens_days_in_advance", int, coerce=_coerce_party,
                  valid=lambda n: 1 <= n <= 365),
        FieldSpec("release_time", str, grounded=True, max_len=40),
        # Not grounded: a small closed vocabulary, and the model reliably maps
        # "Eastern Time" -> "ET"; grounding it just drops the normalized form.
        FieldSpec("release_timezone", str, max_len=40),
        FieldSpec("rolling", bool, coerce=bool),
        FieldSpec("confidence", float, coerce=_coerce_confidence),
        FieldSpec("source_quote", str, grounded=True, max_len=400),
        FieldSpec("notes", str, max_len=300),
    ),
    finalize=_finalize_release,
)


def extract_release_policy(llm: Optional[ReservationLLM],
                           evidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """LLM extraction of a venue's reservation-release policy over web/Reddit
    evidence. Returns {opens_days_in_advance, release_time, release_timezone,
    rolling, confidence, source_quote, notes} or None when not grounded."""
    if llm is None:
        return None
    guard(LLM_CLASSIFY_SINK, evidence)
    evidence_json = json.dumps(evidence, ensure_ascii=False, default=str)
    result = run_task(llm, RELEASE_POLICY_TASK, evidence_json, evidence=evidence_json)
    return result.values if result.from_llm else None


# ------------------------------------------------------------------------ email drafting

_DRAFT_SYSTEM = """You draft a short, polite reservation-request email on behalf of a guest.
Reply with ONLY a JSON object: {"subject": "...", "body": "..."}.
Plain text body, no markdown. Include the party size, date, time (and flexibility if
given), the guest's name, and their callback phone if provided. Ask them to confirm
availability or suggest the nearest alternative. Never mention payment card details."""

DRAFT_TASK = LLMTask(
    name="email_draft",
    system_prompt=_DRAFT_SYSTEM,
    sink=LLM_TEXT_SINK,
    max_tokens=600,
    fields=(
        FieldSpec("subject", str, required=True, max_len=200),
        FieldSpec("body", str, required=True, max_len=4000),
    ),
)


def make_llm_drafter(llm: ReservationLLM, fallback) -> "callable":
    """Wraps the draft task into the EmailChannel drafter signature, falling
    back to the template drafter on any failure."""

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
        result = run_task(llm, DRAFT_TASK, facts)
        if result.from_llm:
            return result.values["subject"], result.values["body"]
        return fallback(slots, decision, instruction=instruction)

    return drafter
