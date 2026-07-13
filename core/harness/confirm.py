"""
Deterministic confirmation parsing (harness spec §9).

The approval path never involves an LLM: a user's reply at a gate is parsed by
keyword sets into YES / NO / EDIT / UNCLEAR. Editable gates (an email draft)
treat unrecognized input as an edit instruction; strict gates (booking, call,
untrusted code) treat it as UNCLEAR — the workflow re-asks instead of
cancelling, and an UNCLEAR can never approve anything.

Safety rule: a negation ANYWHERE in the reply beats an embedded yes-phrase.
"No, don't book it" and "please don't book it" must both parse as NO even
though they contain "book it". Mixed or odd phrasings fall to UNCLEAR/EDIT —
never to YES.
"""

from __future__ import annotations

import re
from enum import Enum

AFFIRMATIVE = {
    "yes", "y", "yeah", "yep", "yup", "correct", "confirm", "confirmed", "go ahead",
    "do it", "book it", "please do", "sounds good", "ok", "okay", "sure", "send it",
    "yes please", "absolutely", "of course", "why not", "sure why not",
}
AFFIRMATIVE_STARTS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm",
                      "absolutely"}
AFFIRMATIVE_PHRASES = ("go ahead", "book it", "do it", "please do", "sounds good",
                       "send it")

# Single tokens that signal refusal wherever they appear.
NEGATIVE_TOKENS = {"no", "n", "nope", "nah", "don't", "dont", "never", "cancel",
                   "stop", "not"}
NEGATIVE_PHRASES = ("do not", "no thanks", "no thank you", "never mind",
                    "nevermind", "forget it", "scrap it", "hold off", "leave it")


class ConfirmDecision(Enum):
    YES = "yes"
    NO = "no"
    EDIT = "edit"
    UNCLEAR = "unclear"


def parse_confirmation(text: str, *, editable: bool = False) -> ConfirmDecision:
    # Normalize: lowercase, strip all punctuation so "No," can't dodge the
    # negation check the way a trailing-comma token used to.
    t = re.sub(r"[^a-z'\s]+", " ", text.strip().lower())
    t = re.sub(r"\s+", " ", t).strip()
    tokens = t.split()
    if not tokens:
        return ConfirmDecision.EDIT if editable else ConfirmDecision.UNCLEAR

    if t in AFFIRMATIVE:
        return ConfirmDecision.YES

    # Negations win over embedded yes-phrases ("no, don't book it",
    # "please don't book it") no matter where they appear.
    if any(tok in NEGATIVE_TOKENS for tok in tokens) or any(p in t for p in NEGATIVE_PHRASES):
        return ConfirmDecision.NO

    if tokens[0] in AFFIRMATIVE_STARTS or any(p in t for p in AFFIRMATIVE_PHRASES):
        return ConfirmDecision.YES
    return ConfirmDecision.EDIT if editable else ConfirmDecision.UNCLEAR
