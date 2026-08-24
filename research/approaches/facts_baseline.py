"""Facts-store baseline: the study's 'existing memory' comparator for arm A.

Retrieves from production's facts table (memory/store.py) with the same
keyword-ish gating the production ContextBuilder uses, formatted as a system
block. Read-only against ~/.friday/memory.db — the baseline must reflect what
production actually knows, not a copy.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from memory.store import FridayStore

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "what", "when", "where", "which", "about", "should", "could", "would",
    "have", "this", "that", "with", "from", "make", "give", "tell", "please",
}
MAX_FACTS = 6


def system_block_for(query: str, store: Optional[FridayStore] = None) -> str:
    """<known_facts> block from keyword hits in the facts table, or ''."""
    try:
        store = store or FridayStore()
        words = [w for w in re.findall(r"[a-zA-Z]{4,}", query.lower())
                 if w not in _STOPWORDS]
        seen: dict[str, str] = {}
        for word in words:
            for key, value, _category in store.search_facts(word):
                seen.setdefault(key, value)
        if not seen:
            return ""
        lines = "\n".join(f"- {k}: {v}" for k, v in list(seen.items())[:MAX_FACTS])
        return f"<known_facts>\n{lines}\n</known_facts>"
    except Exception:
        logger.warning("facts baseline retrieval failed", exc_info=True)
        return ""
