"""Implicit-feedback miners: pure functions over exchange rows.

Run by the nightly harvest stage. Each miner takes chronologically ordered
exchange dicts (id, ts, user_id, user_text — as returned by
ResearchStore.exchanges_since) and returns Signal tuples to insert as
`feedback` rows. No I/O here so the heuristics stay trivially testable.
"""

from __future__ import annotations

import difflib
from typing import NamedTuple

# Mirrors the spirit of FactExtractor.is_correction but scoped to openers —
# "no", "wrong", "actually" mid-sentence are too noisy as signals.
_CORRECTION_OPENERS = (
    "no,", "no.", "no ", "nope", "wrong", "that's wrong", "thats wrong",
    "not what i", "actually,", "actually ", "i said", "i meant",
    "you're wrong", "youre wrong", "incorrect",
)

_REPHRASE_WINDOW_S = 180.0
_REPHRASE_SIMILARITY = 0.6


class Signal(NamedTuple):
    exchange_id: int
    signal: int  # +1 / -1
    source: str
    details: str


def mine_rephrases(exchanges: list[dict]) -> list[Signal]:
    """A quick, similar re-ask means the first answer missed: -1 on the first.

    Consecutive exchanges from the same user within the window whose user texts
    are similar but not identical (identical repeats are usually impatience or
    delivery failure, not a judgment on the answer).
    """
    signals = []
    for prev, cur in zip(exchanges, exchanges[1:]):
        if prev.get("user_id") != cur.get("user_id"):
            continue
        if cur["ts"] - prev["ts"] > _REPHRASE_WINDOW_S:
            continue
        a = (prev["user_text"] or "").strip().lower()
        b = (cur["user_text"] or "").strip().lower()
        if not a or not b or a == b:
            continue
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        if ratio > _REPHRASE_SIMILARITY:
            signals.append(Signal(
                exchange_id=prev["id"],
                signal=-1,
                source="miner:rephrase",
                details=f"similarity={ratio:.2f}",
            ))
    return signals


def mine_corrections(exchanges: list[dict]) -> list[Signal]:
    """A next message opening with a correction marker means the previous
    reply was wrong: -1 on the previous exchange."""
    signals = []
    for prev, cur in zip(exchanges, exchanges[1:]):
        if prev.get("user_id") != cur.get("user_id"):
            continue
        text = (cur["user_text"] or "").strip().lower()
        if any(text.startswith(marker) for marker in _CORRECTION_OPENERS):
            signals.append(Signal(
                exchange_id=prev["id"],
                signal=-1,
                source="miner:correction",
                details=f"opener={text[:40]!r}",
            ))
    return signals


def mine_all(exchanges: list[dict]) -> list[Signal]:
    """All miners over one exchange list, deduplicated per (exchange, source)."""
    seen: set[tuple[int, str]] = set()
    out: list[Signal] = []
    for sig in [*mine_rephrases(exchanges), *mine_corrections(exchanges)]:
        key = (sig.exchange_id, sig.source)
        if key not in seen:
            seen.add(key)
            out.append(sig)
    return out
