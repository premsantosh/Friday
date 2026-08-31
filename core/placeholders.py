"""Placeholder identity: the model refers to the user's stored identity
through {{tokens}} without ever seeing the values.

Extends core/profile.py's contract to chat. The system prompt advertises
which tokens exist (labels only, via the same no-values projection that
`descriptors()` provides); the model writes e.g. "You are {{full_name}},
sir"; `resolve()` substitutes the real value at the delivery boundary —
the outgoing Telegram/terminal/TTS string. Everything persisted (memory
turns, langgraph checkpoints, response cache, research exchanges) keeps
the placeholder form, because all of it gets replayed into later prompts.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

from core.profile import FIELD_BY_KEY, FORMAL_CONTEXT, UserProfile

logger = logging.getLogger(__name__)

# Profile keys the model may reference. Deliberately excludes phone,
# address, date_of_birth and insurance_* — those exist for form-filling
# and have no business in a spoken reply.
PROFILE_TOKENS = ("full_name", "first_name", "last_name", "email")

# Tokens backed by the auto-extracted facts table: token -> candidate keys,
# first hit wins. Facts are extractor-written, so require decent confidence.
FACT_TOKENS: Dict[str, tuple] = {
    "spouse_name": ("spouse_name", "wife_name", "husband_name", "partner_name"),
}
FACT_MIN_CONFIDENCE = 0.6

TOKEN_RE = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")


class PlaceholderResolver:
    """Construct per use; the lookups are two small local sqlite reads.

    `store` (a FridayStore) is optional: without it the fact-backed tokens
    (spouse_name) simply aren't available.
    """

    def __init__(self, profile: Optional[UserProfile] = None, store=None):
        self._profile = profile
        self._store = store
        self._tokens: Optional[Dict[str, str]] = None

    def tokens(self) -> Dict[str, str]:
        """token -> real value. LOCAL ONLY: never into a prompt or egress sink."""
        if self._tokens is not None:
            return self._tokens
        out: Dict[str, str] = {}
        try:
            profile = self._profile or UserProfile()
            # Formal context: the canonical legal identity, not the casual
            # persona used for restaurant bookings.
            out.update(profile.values(PROFILE_TOKENS, context=FORMAL_CONTEXT))
        except Exception:
            logger.warning("placeholder profile lookup failed", exc_info=True)
        if self._store is not None:
            try:
                personal = self._store.recall_by_category(
                    "personal", min_confidence=FACT_MIN_CONFIDENCE)
                for token, candidates in FACT_TOKENS.items():
                    for key in candidates:
                        if personal.get(key):
                            out[token] = personal[key]
                            break
            except Exception:
                logger.warning("placeholder fact lookup failed", exc_info=True)
        self._tokens = out
        return out

    def identity_block(self) -> str:
        """System-prompt section: available tokens with labels, never values."""
        toks = self.tokens()
        if not toks:
            return ""
        described = []
        for token in (*PROFILE_TOKENS, *FACT_TOKENS):
            if token not in toks:
                continue
            field = FIELD_BY_KEY.get(token)
            label = field.label.lower() if field else token.replace("_", " ")
            described.append(f"{{{{{token}}}}} ({label})")
        return (
            "USER IDENTITY:\n"
            "- You know the following about the user, held as placeholder tokens: "
            + ", ".join(described) + ".\n"
            '- To say one, write the token exactly as shown (e.g. "You are '
            '{{full_name}}, sir"). The real value is substituted locally after '
            "you reply; you never see it, and that is by design — do not ask "
            "the user for these, and do not say you don't know them.\n"
            "- Use tokens only in your spoken reply, never inside tool arguments.\n"
            "- Anything not listed here, you genuinely do not know."
        )

    def resolve(self, text: str) -> str:
        """Substitute {{token}} with real values. Delivery boundary only.

        Idempotent: resolved text contains no tokens, so a second pass is a
        no-op. Unknown tokens are stripped with a warning.
        """
        if not text or "{{" not in text:
            return text
        toks = self.tokens()

        def _sub(match: re.Match) -> str:
            token = match.group(1)
            if token in toks:
                return toks[token]
            logger.warning("Unknown placeholder token {{%s}} stripped from reply", token)
            return ""

        resolved = TOKEN_RE.sub(_sub, text)
        if resolved != text:
            resolved = re.sub(r"[ \t]{2,}", " ", resolved).strip()
        return resolved


def identity_block(store=None) -> str:
    """Module convenience for prompt assembly; never raises."""
    try:
        return PlaceholderResolver(store=store).identity_block()
    except Exception:
        logger.warning("identity block generation failed", exc_info=True)
        return ""
