"""Deterministic style-compliance checks (free, no LLM).

These encode Jarvis's fixed persona rules; used as cheap metrics per response
and as the catastrophic-forgetting canary gating arm-B adapters: a fine-tune
that stops honoring the persona fails here before any judge is paid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_SENTENCES = 4  # persona says ~3 for simple requests; allow slack for judged prompts


@dataclass
class StyleReport:
    sentences: int
    exclamations: int
    addresses_user: bool
    chars: int

    @property
    def compliant(self) -> bool:
        return self.exclamations == 0 and self.sentences <= MAX_SENTENCES


def check_style(text: str) -> StyleReport:
    stripped = text.strip()
    # Sentence count: terminal punctuation runs (won't be fooled by "e.g." much,
    # good enough for a canary metric).
    sentences = len(re.findall(r"[.!?]+(?:\s|$)", stripped)) or (1 if stripped else 0)
    return StyleReport(
        sentences=sentences,
        exclamations=stripped.count("!"),
        addresses_user=bool(re.search(r"\bsir\b", stripped, re.IGNORECASE)),
        chars=len(stripped),
    )


def style_score(texts: list[str]) -> float:
    """Fraction of responses passing the hard style rules (0..1)."""
    if not texts:
        return 0.0
    return sum(1 for t in texts if check_style(t).compliant) / len(texts)
