"""
Egress enforcement — every boundary where data leaves the process is a
declared Sink, checked deterministically before anything is sent
(harness spec §5).

Two modes:
  - ALLOWLIST: only the declared fields may appear in the payload; an unknown
    field is a violation (data minimization by construction — e.g. discovery
    queries may carry only business name + location, never user PII).
  - SCAN: free-form payloads (prompts, email bodies) are scanned for things
    that must never leave: card numbers (Luhn-validated), CVVs next to their
    label, and secret-shaped values.

Violations **block and fail the step** (`EgressViolation`); nothing is
silently redacted on the way out. Redaction exists only for *logs*
(`install_log_redaction`), where masking beats losing the log line.

Also here: `purge_slots`, the terminal-session PII purge used by
`ConversationalWorkflow.on_terminal`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Tuple, Union

logger = logging.getLogger(__name__)


class SinkMode(Enum):
    ALLOWLIST = "allowlist"
    SCAN = "scan"


@dataclass(frozen=True)
class Sink:
    name: str
    mode: SinkMode
    allowed_fields: frozenset = frozenset()


class EgressViolation(Exception):
    def __init__(self, sink: str, reason: str):
        self.sink = sink
        self.reason = reason
        super().__init__(f"egress blocked at sink {sink!r}: {reason}")


# ------------------------------------------------------------------ scanners

# 13–19 digits, optionally separated by spaces/dashes — candidate card number.
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)")
_CVV_RE = re.compile(r"(?i)\b(?:cvv|cvc|cvv2|security code)\b\D{0,8}\d{3,4}")
_SECRET_KEY_RE = re.compile(r"(?i)(api[_-]?key|secret|password|token|authorization|credential)")
_SECRET_VALUE_RE = re.compile(r"(?:sk-|key-|Bearer\s+)[A-Za-z0-9_\-]{16,}")
# Phone-ish field names: their digits are intentional egress, skip the card scan.
_PHONE_KEY_RE = re.compile(r"(?i)phone|callback")


def luhn_ok(digits: str) -> bool:
    """Luhn checksum over a digit string (card-number validity)."""
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


_luhn_ok = luhn_ok  # internal alias


def _scan_text(sink: str, text: str, *, skip_card: bool = False) -> None:
    if not skip_card:
        for m in _CARD_RE.finditer(text):
            digits = re.sub(r"[ \-]", "", m.group(0))
            if 13 <= len(digits) <= 19 and _luhn_ok(digits):
                raise EgressViolation(sink, "payment-card-shaped number in payload")
    if _CVV_RE.search(text):
        raise EgressViolation(sink, "CVV-like value in payload")
    if _SECRET_VALUE_RE.search(text):
        raise EgressViolation(sink, "secret-shaped value in payload")


def _walk(obj: Any, key: str = "") -> Iterator[Tuple[str, str]]:
    """Yield (nearest_key, text) for every leaf string/number in a payload."""
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield from _walk(v, str(k))
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _walk(v, key)
    elif obj is not None:
        yield key, str(obj)


def guard(sink: Sink, payload: Union[str, Mapping, Iterable]) -> Union[str, Mapping, Iterable]:
    """Validate an outbound payload against the sink's policy. Returns the
    payload unchanged, or raises EgressViolation (block-and-fail)."""
    if sink.mode == SinkMode.ALLOWLIST:
        if not isinstance(payload, Mapping):
            raise EgressViolation(sink.name, "allowlist sinks take a mapping payload")
        unknown = set(payload.keys()) - set(sink.allowed_fields)
        if unknown:
            raise EgressViolation(
                sink.name, f"fields not allowed for this sink: {sorted(unknown)}")

    if isinstance(payload, str):
        _scan_text(sink.name, payload)
    else:
        for key, text in _walk(payload):
            if _SECRET_KEY_RE.search(key) and text:
                raise EgressViolation(sink.name, f"secret-named field {key!r} in payload")
            _scan_text(sink.name, text, skip_card=bool(_PHONE_KEY_RE.search(key)))
    return payload


# ------------------------------------------------------------- log redaction

_EMAIL_TEXT_RE = re.compile(r"\b([\w.+-])[\w.+-]*@[\w-]+\.[\w.-]+\b")
_PHONE_TEXT_RE = re.compile(r"(?<![\d\w])\+?\d[\d\- ().]{7,}(\d{4})(?!\d)")


def redact_text(text: str) -> str:
    """Deterministic masking for log lines: card numbers (Luhn), e-mail
    addresses, and phone numbers (last four kept)."""

    def _mask_card(m: re.Match) -> str:
        digits = re.sub(r"[ \-]", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return "████CARD████"
        return m.group(0)

    text = _CARD_RE.sub(_mask_card, text)
    text = _EMAIL_TEXT_RE.sub(r"\1***@***", text)
    text = _PHONE_TEXT_RE.sub(r"***\1", text)
    return text


class RedactionFilter(logging.Filter):
    """Masks PII in the rendered message. Attach to *handlers* (filters on a
    logger don't apply to records propagated from child loggers)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
            redacted = redact_text(rendered)
            if redacted != rendered:
                record.msg = redacted
                record.args = ()
        except Exception:  # never let redaction break logging itself
            pass
        return True


def install_log_redaction() -> None:
    """Attach the redaction filter to every current root handler — and to the
    stderr last-resort handler, which is what emits when nothing is configured.
    Call after logging is configured (main.py). Idempotent."""
    handlers = list(logging.getLogger().handlers)
    if logging.lastResort is not None:
        handlers.append(logging.lastResort)
    for handler in handlers:
        if not any(isinstance(f, RedactionFilter) for f in handler.filters):
            handler.addFilter(RedactionFilter())


# ----------------------------------------------------------------- PII purge

def purge_slots(session, keys: Iterable[str]) -> None:
    """Blank the listed PII slots on a finished session (spec §8 L2). The
    booking facts (business, date, time, confirmation) are kept."""
    for key in keys:
        if key in session.slots:
            session.slots[key] = None
