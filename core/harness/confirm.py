"""
Deterministic confirmation parsing (harness spec §9).

The approval path never involves an LLM: a user's reply at a gate is parsed by
keyword sets into YES / NO / EDIT / UNCLEAR. Editable gates (an email draft)
treat unrecognized input as an edit instruction; strict gates (booking, call,
untrusted code) treat it as UNCLEAR — the workflow re-asks instead of
cancelling, and an UNCLEAR can never approve anything.
"""

from __future__ import annotations

from enum import Enum

AFFIRMATIVE = {
    "yes", "y", "yeah", "yep", "yup", "correct", "confirm", "confirmed", "go ahead",
    "do it", "book it", "please do", "sounds good", "ok", "okay", "sure", "send it",
    "yes please", "absolutely", "of course",
}
AFFIRMATIVE_STARTS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm",
                      "absolutely"}
AFFIRMATIVE_PHRASES = ("go ahead", "book it", "do it", "please do", "sounds good",
                       "send it")

NEGATIVE = {
    "no", "n", "nope", "nah", "don't", "do not", "cancel", "stop", "no thanks",
    "no thank you", "never mind", "nevermind", "forget it", "scrap it", "hold off",
    "not now", "leave it",
}
NEGATIVE_STARTS = {"no", "nope", "nah", "don't", "dont", "never", "cancel", "stop", "not"}


class ConfirmDecision(Enum):
    YES = "yes"
    NO = "no"
    EDIT = "edit"
    UNCLEAR = "unclear"


def parse_confirmation(text: str, *, editable: bool = False) -> ConfirmDecision:
    t = text.strip().lower().rstrip("!.,")
    first = t.split()[0] if t else ""

    if t in AFFIRMATIVE:
        return ConfirmDecision.YES
    # Negations win over embedded yes-phrases ("don't book it").
    if t in NEGATIVE or first in NEGATIVE_STARTS:
        return ConfirmDecision.NO
    if first in AFFIRMATIVE_STARTS or any(p in t for p in AFFIRMATIVE_PHRASES):
        return ConfirmDecision.YES
    return ConfirmDecision.EDIT if editable else ConfirmDecision.UNCLEAR
