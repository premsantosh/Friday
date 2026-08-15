"""Implicit-feedback miners: pure functions over exchange rows.

Run by the nightly harvest stage. Each miner takes chronologically ordered
exchange dicts (id, ts, user_id, user_text — as returned by
ResearchStore.exchanges_since) and returns Signal tuples to insert as
`feedback` rows. No I/O here so the heuristics stay trivially testable.
"""

from __future__ import annotations

import difflib
import json
from typing import NamedTuple

# Mirrors the spirit of FactExtractor.is_correction but scoped to openers —
# "no", "wrong", "actually" mid-sentence are too noisy as signals.
_CORRECTION_OPENERS = (
    "no,", "no.", "no ", "nope", "wrong", "that's wrong", "thats wrong",
    "not what i", "actually,", "actually ", "i said", "i meant",
    "you're wrong", "youre wrong", "incorrect",
)

# Short, unambiguous approval. Anything longer or hedged ("thanks, but...") is
# handled by the negative-wins rule in mine_all.
_THANKS_OPENERS = (
    "thanks", "thank you", "perfect", "brilliant", "exactly", "that's it",
    "thats it", "nice one", "cheers", "spot on", "lovely", "great, thanks",
)

_REPHRASE_WINDOW_S = 180.0
_REPHRASE_SIMILARITY = 0.6


class Signal(NamedTuple):
    exchange_id: int
    signal: int  # +1 / -1
    source: str
    details: str  # JSON object; always carries followup_id


def parse_details(raw: str | None) -> dict:
    """Signal/feedback details as a dict.

    Details used to be free text ("similarity=0.83"); rows written before this
    change parse to {} rather than blowing up a nightly stage.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


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
                details=json.dumps({"followup_id": cur["id"],
                                    "similarity": round(ratio, 2)}),
            ))
    return signals


def mine_corrections(exchanges: list[dict]) -> list[Signal]:
    """A next message opening with a correction marker means the previous
    reply was wrong: -1 on the previous exchange.

    followup_id is the message that did the correcting. Arm B's correction
    synthesis needs that exact message, and recording it here beats re-deriving
    it later from timestamps and a user_id that may not have been backfilled.
    """
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
                details=json.dumps({"followup_id": cur["id"],
                                    "opener": text[:40]}),
            ))
    return signals


def mine_thanks(exchanges: list[dict]) -> list[Signal]:
    """A quick appreciative follow-up means the answer landed: +1 on the previous.

    The only positive miner. Without it the corpus is negative-only, which biases
    arm C's evolver toward things to avoid and gives arm B's selection nothing to
    weigh its banked negatives against.

    Held to a higher precision bar than the negative miners, because a positive
    puts an exchange straight into arm B's training set while a negative only
    holds one back. "thanks, but that's wrong" opens with thanks and is not
    approval, so a correction marker anywhere in the message disqualifies it.
    """
    signals = []
    for prev, cur in zip(exchanges, exchanges[1:]):
        if prev.get("user_id") != cur.get("user_id"):
            continue
        if cur["ts"] - prev["ts"] > _REPHRASE_WINDOW_S:
            continue
        text = (cur["user_text"] or "").strip().lower()
        if not any(text.startswith(marker) for marker in _THANKS_OPENERS):
            continue
        if any(marker in text for marker in _CORRECTION_OPENERS):
            continue
        signals.append(Signal(
            exchange_id=prev["id"],
            signal=1,
            source="miner:thanks",
            details=json.dumps({"followup_id": cur["id"], "opener": text[:40]}),
        ))
    return signals


def mine_all(exchanges: list[dict]) -> list[Signal]:
    """All miners over one exchange list, deduplicated per (exchange, source).

    Negative wins: "thanks, but actually that's wrong" opens with thanks and
    still trips the correction miner, and it is not approval. Any exchange with
    a negative signal in this batch drops its positive.
    """
    seen: set[tuple[int, str]] = set()
    out: list[Signal] = []
    for sig in [*mine_rephrases(exchanges), *mine_corrections(exchanges),
                *mine_thanks(exchanges)]:
        key = (sig.exchange_id, sig.source)
        if key not in seen:
            seen.add(key)
            out.append(sig)

    negative = {s.exchange_id for s in out if s.signal < 0}
    return [s for s in out if s.signal < 0 or s.exchange_id not in negative]
