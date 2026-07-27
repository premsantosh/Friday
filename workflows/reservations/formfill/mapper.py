"""
Field mapping — decide which fact fills which form field.

The model gets three things: the page's form fields, the *keys and descriptions*
of the facts we hold about the user, and the booking request. It never gets a
single one of those fact values. It answers with a key; the value is substituted
locally in `agent.py`. So the worst a compromised or confused model can do is
put the right data in the wrong box — which the confirmation gate shows the user
before anything is submitted — not exfiltrate a date of birth.

Every proposal is validated deterministically, in the spirit of
core/harness/extract.py:

  - a `ref` must be one the harvester actually produced,
  - a `fact_key` must be in the closed profile registry *and* be a fact we hold,
  - a literal for a select/radio must be copied from that field's own options,
  - a field whose label is plainly asking for personal data may not be filled
    with a model-authored literal at all — that's the anti-fabrication rule, and
    it's enforced here rather than merely requested in the prompt.

With no ANTHROPIC_API_KEY the deterministic matcher in `_fallback_map` runs
instead: label matching against the same profile registry. Degraded, not broken.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from core.harness import FieldSpec, LLMTask, Sink, SinkMode, run_task
from core.profile import PROFILE_FIELDS, PROFILE_KEYS, match_key

from .harvest import FormField, FormSnapshot

logger = logging.getLogger(__name__)

# Only these four keys may cross the boundary. `available_facts` carries
# descriptors (key/label/description) — never values; `request` carries booking
# facts with the guest's name, phone and email deliberately left out.
FORM_MAP_SINK = Sink("llm.formmap", SinkMode.ALLOWLIST,
                     frozenset({"page", "fields", "available_facts", "request"}))

# Fields whose label plainly asks for personal data. A model-authored literal
# here would be a fabricated identity, so we refuse it outright: it comes from
# the profile or it goes on the missing list.
_PERSONAL_KEYS = frozenset({
    "full_name", "first_name", "last_name", "email", "phone",
    "street_address", "city", "state", "postal_code",
    "date_of_birth", "insurance_provider", "insurance_member_id", "employer",
})

_VOCABULARY = ", ".join(f.key for f in PROFILE_FIELDS)

_SYSTEM = f"""You map the fields of a web form to the data available to fill it in.

You are given the page, its form fields, and the facts that are AVAILABLE about
the person filling it in. You see fact KEYS and descriptions only — never their
values. Name the key; the value is filled in locally.

Reply with ONLY a JSON object:
{{"mappings": [{{"ref": "<field ref, copied exactly>",
                "source": "fact" | "literal" | "skip",
                "fact_key": "<an available key>" or null,
                "literal": "<the text to type>" or null,
                "confidence": 0.0-1.0}}],
  "missing_facts": [{{"ref": "...", "label": "...", "suggested_key": "<key>"}}],
  "notes": "one short sentence"}}

Rules:
- One mapping entry per form field. Use "skip" for anything that shouldn't be filled.
- "fact" when an available fact answers the field. Never write the fact's value.
- "literal" only for text derivable from the REQUEST section (the reason for the
  visit, preferred date/time, notes) or for choosing among a field's options.
- For a select or radio field, "literal" MUST be copied exactly from that field's
  own options list.
- For a checkbox, "literal" is "true" or "false". Tick a required consent or terms
  box; leave marketing, newsletter and SMS opt-ins "false".
- NEVER invent a person's name, email, phone, address, date of birth or any other
  personal detail as a literal. Personal data is a fact key, or it is missing.
- A required field you cannot fill from an available fact goes in missing_facts,
  with the closest key from this vocabulary: {_VOCABULARY}"""


@dataclass
class Assignment:
    """One field we intend to fill, and where the value comes from."""
    ref: str
    source: str                      # "fact" | "literal"
    fact_key: Optional[str] = None
    literal: Optional[str] = None
    option_value: Optional[str] = None   # resolved <option>/radio value to select

    def to_dict(self) -> Dict[str, Any]:
        return {"ref": self.ref, "source": self.source, "fact_key": self.fact_key,
                "literal": self.literal, "option_value": self.option_value}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Assignment":
        return cls(ref=d["ref"], source=d.get("source", "literal"),
                   fact_key=d.get("fact_key"), literal=d.get("literal"),
                   option_value=d.get("option_value"))


@dataclass
class MissingFact:
    ref: str
    label: str
    suggested_key: Optional[str] = None
    # A fixed menu, when the field has one. Asking "what shall I put?" about a
    # dropdown invites an answer that matches none of its options.
    options: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ref": self.ref, "label": self.label,
                "suggested_key": self.suggested_key, "options": self.options}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MissingFact":
        return cls(ref=d["ref"], label=d.get("label", ""),
                   suggested_key=d.get("suggested_key"),
                   options=list(d.get("options") or []))


@dataclass
class FieldMap:
    assignments: List[Assignment] = field(default_factory=list)
    missing: List[MissingFact] = field(default_factory=list)
    provenance: str = "none"         # "llm" | "fallback" | "none"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"assignments": [a.to_dict() for a in self.assignments],
                "missing": [m.to_dict() for m in self.missing],
                "provenance": self.provenance, "notes": self.notes}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FieldMap":
        return cls(assignments=[Assignment.from_dict(a) for a in d.get("assignments") or []],
                   missing=[MissingFact.from_dict(m) for m in d.get("missing") or []],
                   provenance=d.get("provenance", "none"), notes=d.get("notes", ""))


# --------------------------------------------------------------------- validation

def match_option(fld: FormField, literal: str) -> Optional[str]:
    """Resolve a proposed choice to a real option value, or None if off-menu."""
    want = (literal or "").strip().lower()
    if not want:
        return None
    for option in fld.options:
        if option.label.strip().lower() == want or option.value.strip().lower() == want:
            return option.value
    for option in fld.options:                      # allow a contained match
        label = option.label.strip().lower()
        if label and (want in label or label in want):
            return option.value
    return None


def _truthy(literal: str) -> Optional[bool]:
    low = (literal or "").strip().lower()
    if low in ("true", "yes", "y", "1", "checked", "on"):
        return True
    if low in ("false", "no", "n", "0", "unchecked", "off"):
        return False
    return None


def _personal_key_for(fld: FormField) -> Optional[str]:
    """The profile key this field is plainly asking for, if it's personal data."""
    key = match_key(f"{fld.label} {fld.placeholder} {fld.name}")
    return key if key in _PERSONAL_KEYS else None


def _validate(raw_mappings: Sequence[Any], raw_missing: Sequence[Any],
              snapshot: FormSnapshot, available: Set[str]) -> tuple:
    """Turn the model's proposals into assignments we're willing to execute."""
    assignments: List[Assignment] = []
    missing: List[MissingFact] = []
    seen: Set[str] = set()

    for item in raw_mappings or []:
        if not isinstance(item, dict):
            continue
        ref = item.get("ref")
        fld = snapshot.by_ref(ref) if isinstance(ref, str) else None
        if fld is None or ref in seen:
            continue                       # a ref the harvester never produced
        seen.add(ref)
        source = str(item.get("source") or "").strip().lower()
        if source == "skip":
            continue

        personal = _personal_key_for(fld)

        if source == "fact":
            key = item.get("fact_key")
            if not isinstance(key, str) or key not in PROFILE_KEYS:
                continue                   # outside the closed registry
            if key not in available:
                missing.append(MissingFact(ref=ref, label=fld.label, suggested_key=key))
                continue
            assignments.append(Assignment(ref=ref, source="fact", fact_key=key))
            continue

        if source != "literal":
            continue
        literal = item.get("literal")
        if not isinstance(literal, str) or not literal.strip() or len(literal) > 2000:
            continue

        if personal:
            # Refused: a literal here would be an invented identity. Use the
            # real fact if we have it, otherwise say we're missing it.
            if personal in available:
                assignments.append(Assignment(ref=ref, source="fact", fact_key=personal))
            else:
                missing.append(MissingFact(ref=ref, label=fld.label, suggested_key=personal))
            continue

        if fld.describes_choice:
            value = match_option(fld, literal)
            if value is None:
                continue                   # off-menu choice: drop it
            assignments.append(Assignment(ref=ref, source="literal",
                                          literal=literal.strip(), option_value=value))
            continue

        if fld.type == "checkbox":
            state = _truthy(literal)
            if state is None:
                continue
            assignments.append(Assignment(ref=ref, source="literal",
                                          literal="true" if state else "false"))
            continue

        assignments.append(Assignment(ref=ref, source="literal", literal=literal.strip()))

    assigned = {a.ref for a in assignments}
    flagged = {m.ref for m in missing}
    for item in raw_missing or []:
        if not isinstance(item, dict):
            continue
        ref = item.get("ref")
        fld = snapshot.by_ref(ref) if isinstance(ref, str) else None
        if fld is None or ref in assigned or ref in flagged:
            continue
        key = item.get("suggested_key")
        missing.append(MissingFact(
            ref=ref, label=fld.label,
            suggested_key=key if isinstance(key, str) and key in PROFILE_KEYS else None))
        flagged.add(ref)

    # Backstop, independent of whatever the model said. Two kinds of gap:
    #
    #  - a field we have vocabulary for but no value (date of birth, insurance
    #    carrier). Worth asking even when the form marks it optional: a clinic
    #    that prints an insurance box will chase you for it later, and the
    #    answer is stored once and reused forever.
    #  - a required field nobody accounted for at all.
    for fld in snapshot.fields:
        if fld.ref in assigned or fld.ref in flagged:
            continue
        key = match_key(f"{fld.label} {fld.placeholder} {fld.name}")
        if (key and key not in available) or fld.required:
            missing.append(MissingFact(
                ref=fld.ref, label=fld.label or fld.ref, suggested_key=key,
                options=[o.label for o in fld.options if o.value]))
            flagged.add(fld.ref)
    return assignments, missing


def _map_task(snapshot: FormSnapshot, available: Set[str]) -> LLMTask:
    """The task, with per-item validation closed over this page's fields."""

    def finalize(values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        assignments, missing = _validate(
            values.get("mappings") or [], values.get("missing_facts") or [],
            snapshot, available)
        if not assignments and not missing:
            return None                    # nothing usable → caller's fallback
        return {"assignments": assignments, "missing": missing,
                "notes": values.get("notes", "")}

    return LLMTask(
        name="form_field_mapping",
        system_prompt=_SYSTEM,
        sink=FORM_MAP_SINK,
        max_tokens=2000,
        fields=(
            FieldSpec("mappings", list, required=True, valid=lambda v: isinstance(v, list)),
            FieldSpec("missing_facts", list, valid=lambda v: isinstance(v, list)),
            FieldSpec("notes", str, max_len=300),
        ),
        finalize=finalize,
    )


# ------------------------------------------------------------------ public entry

def map_fields(llm, snapshot: FormSnapshot, descriptors: List[Dict[str, str]],
               request: Dict[str, Any]) -> FieldMap:
    """Decide what goes in each field.

    `descriptors` is `UserProfile.descriptors()` — keys and labels, no values.
    `request` is the booking facts; the caller must leave the guest's name,
    phone and email out of it (they arrive as fact keys instead).
    """
    available = {d["key"] for d in descriptors}
    payload = {
        "page": {"url": snapshot.url, "heading": snapshot.heading},
        "fields": [f.describe() for f in snapshot.fields],
        "available_facts": descriptors,
        "request": {k: v for k, v in (request or {}).items() if v not in (None, "")},
    }

    def fallback() -> Optional[Dict[str, Any]]:
        assignments, missing = _fallback_map(snapshot, available, request)
        return {"assignments": assignments, "missing": missing,
                "notes": "matched deterministically by field label"}

    result = run_task(llm, _map_task(snapshot, available), payload, fallback=fallback)
    if result.values is None:
        return FieldMap(provenance="none")
    return FieldMap(assignments=result.values["assignments"],
                    missing=result.values["missing"],
                    provenance=result.provenance,
                    notes=result.values.get("notes", ""))


# ------------------------------------------------------------- deterministic path

# Request slots that can answer a form field, by the label fragments that ask
# for them. Kept in the same spirit as the old GenericWebChannel hints.
_REQUEST_HINTS = (
    ("service_type", ("service", "reason", "treatment", "what brings you",
                      "appointment type", "type of visit", "concern", "injury")),
    ("special_requests", ("message", "notes", "comments", "details", "tell us",
                          "additional", "anything else", "description")),
    ("date", ("date", "preferred day", "day")),
    ("time", ("time", "preferred time")),
    ("party_size", ("party", "guests", "people", "how many", "covers")),
)


def _fallback_map(snapshot: FormSnapshot, available: Set[str],
                  request: Dict[str, Any]):
    """Label matching against the profile registry, for when there's no LLM."""
    assignments: List[Assignment] = []
    missing: List[MissingFact] = []

    for fld in snapshot.fields:
        haystack = f"{fld.label} {fld.placeholder} {fld.name}".lower()

        key = match_key(haystack)
        if key:
            if key in available:
                assignments.append(Assignment(ref=fld.ref, source="fact", fact_key=key))
            elif fld.required:
                missing.append(MissingFact(ref=fld.ref, label=fld.label, suggested_key=key))
            continue

        value = next((request.get(slot) for slot, hints in _REQUEST_HINTS
                      if any(h in haystack for h in hints) and request.get(slot)), None)
        if value is not None and not fld.describes_choice and fld.type != "checkbox":
            assignments.append(Assignment(ref=fld.ref, source="literal", literal=str(value)))
            continue

        if fld.type == "checkbox" and fld.required:
            # A required checkbox is a terms/consent box; marketing opt-ins
            # aren't required. Ticking it is what the user asked us to do.
            assignments.append(Assignment(ref=fld.ref, source="literal", literal="true"))
            continue

        if fld.required:
            missing.append(MissingFact(ref=fld.ref, label=fld.label,
                                       suggested_key=_personal_key_for(fld)))
    return assignments, missing
