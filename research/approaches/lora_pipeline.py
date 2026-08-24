"""Arm B data pipeline: conversations + feedback -> SFT dataset for mlx_lm.

Selection rules (from the study protocol):
  * route='chat' exchanges only.
  * Explicit 👎 (or any net-negative feedback): excluded from SFT, banked for
    phase-2 preference tuning.
  * Explicit 👍: included immediately.
  * Unflagged exchanges: included once older than 48 h — silence long enough
    counts as weak approval (the miners have had time to flag rephrases).
  * Mined corrections: the correction message tells us what the reply should
    have been; a strong model synthesizes the corrected reply once per
    correction (cached by exchange id), giving a positive example instead of
    just losing the pair.

Every example uses the pinned static persona prompt — never the live
date-stamped production prompt, which would teach the model a frozen clock.
The final dataset mixes personal examples 1:1 with a pinned general-instruct
replay slice; the valid split is by stable hash so examples never migrate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Callable, Optional

from research.miners import parse_details

from research.db import ResearchStore
from research.persona import PERSONA_PROMPT

logger = logging.getLogger(__name__)

REPLAY_PATH = Path(__file__).parent.parent / "data" / "replay" / "replay.jsonl"
NEUTRAL_MIN_AGE_S = 48 * 3600
VALID_FRACTION_MOD = 10  # hash % 10 == 0 -> valid (~10%)
SEED = 42

_CORRECTION_SYNTH_PROMPT = """A butler-style assistant (persona: address the user as "sir", deadpan, no exclamation marks, <=3 sentences for simple requests) gave a reply the user then corrected.

User message: {user_text}
Assistant's reply (wrong): {reply_text}
User's correction: {correction_text}

Write only the reply the assistant SHOULD have given, incorporating the correction, in persona."""


def _example(user_text: str, reply_text: str) -> dict:
    return {"messages": [
        {"role": "system", "content": PERSONA_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply_text},
    ]}


def _is_valid_split(user_text: str) -> bool:
    digest = hashlib.sha256(user_text.encode()).digest()
    return digest[0] % VALID_FRACTION_MOD == 0


def net_signal(store: ResearchStore, exchange_id: int) -> Optional[int]:
    """Sum of feedback signals, or None when unflagged."""
    rows = store.feedback_for(exchange_id)
    if not rows:
        return None
    return sum(r["signal"] for r in rows)


def select_personal_examples(store: ResearchStore, *,
                             now: Optional[float] = None) -> dict:
    """Partition all chat exchanges into SFT examples / banked negatives / deferred."""
    now = now if now is not None else time.time()
    included, included_ids, banked_negative, deferred = [], [], [], []
    rows = store.query(
        "SELECT id, ts, user_text, reply_text FROM exchanges"
        " WHERE route = 'chat' ORDER BY id")
    for row in rows:
        signal = net_signal(store, row["id"])
        if signal is not None and signal < 0:
            banked_negative.append(row["id"])
        elif signal is None and now - row["ts"] < NEUTRAL_MIN_AGE_S:
            deferred.append(row["id"])  # too fresh — miners may still flag it
        else:
            included.append(_example(row["user_text"], row["reply_text"]))
            included_ids.append(row["id"])
    # included_ids feeds the artifact's provenance manifest: which exact turns
    # this adapter was trained on.
    return {"included": included, "included_ids": included_ids,
            "banked_negative": banked_negative, "deferred": deferred}


def synthesize_corrections(
    store: ResearchStore,
    cache_path: Path,
    llm_fn: Callable[[str], str],
) -> list[dict]:
    """Corrected-behavior examples for exchanges flagged by miner:correction.

    One paid synthesis per correction ever: results append to cache_path
    (jsonl keyed by exchange_id) and are reused on every later build. Returns
    [{"exchange_id": int, "example": dict}] so callers can record which
    corrections an artifact consumed.
    """
    cache: dict[int, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            row = json.loads(line)
            cache[row["exchange_id"]] = row["example"]

    flagged = store.query(
        "SELECT exchange_id, details FROM feedback WHERE source = 'miner:correction'"
        " ORDER BY exchange_id")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for row in flagged:
        eid = row["exchange_id"]
        if eid in cache:
            store.emit("correction.cache_hit", subject_type="exchange",
                       subject_id=eid, arm="lora")
            continue
        exchange = store.get_exchange(eid)
        if exchange is None:
            continue
        # The miner recorded which message did the correcting. Re-deriving it
        # here used to match on `user_id IS ?`, which matches the wrong follow-up
        # whenever user_id is NULL (record_chat inserts NULL; record_turn
        # backfills it, but only when the dedupe window catches the pair).
        followup_id = parse_details(row["details"]).get("followup_id")
        if followup_id is not None:
            follow_ups = store.query(
                "SELECT user_text FROM exchanges WHERE id = ?", (followup_id,))
        else:
            follow_ups = store.query(
                "SELECT user_text FROM exchanges WHERE id > ? ORDER BY id LIMIT 1",
                (eid,))
        if not follow_ups:
            continue
        follow_up = follow_ups[0]
        try:
            corrected = llm_fn(_CORRECTION_SYNTH_PROMPT.format(
                user_text=exchange["user_text"],
                reply_text=exchange["reply_text"],
                correction_text=follow_up["user_text"],
            )).strip()
        except Exception:
            logger.warning("correction synthesis failed for exchange %s", eid,
                           exc_info=True)
            continue
        if not corrected:
            continue
        example = _example(exchange["user_text"], corrected)
        cache[eid] = example
        with open(cache_path, "a") as f:
            f.write(json.dumps({"exchange_id": eid, "example": example}) + "\n")
        store.emit("correction.synthesized", subject_type="exchange",
                   subject_id=eid, arm="lora",
                   detail={"followup_id": followup_id, "chars": len(corrected)})
    return [{"exchange_id": eid, "example": ex} for eid, ex in sorted(cache.items())]


def load_replay(path: Path = REPLAY_PATH) -> list[dict]:
    examples = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        row["messages"] = [{"role": "system", "content": PERSONA_PROMPT},
                           *row["messages"]]
        examples.append(row)
    return examples


def build_dataset(
    store: ResearchStore,
    out_dir: Path,
    *,
    correction_llm_fn: Optional[Callable[[str], str]] = None,
    corrections_cache: Optional[Path] = None,
    replay_path: Path = REPLAY_PATH,
    now: Optional[float] = None,
) -> dict:
    """Write train.jsonl / valid.jsonl to out_dir; returns build stats."""
    selection = select_personal_examples(store, now=now)
    _emit_selection(store, selection)
    personal = selection["included"]
    correction_ids: list[int] = []
    if correction_llm_fn is not None and corrections_cache is not None:
        corrections = synthesize_corrections(store, corrections_cache,
                                             correction_llm_fn)
        correction_ids = [c["exchange_id"] for c in corrections]
        personal = personal + [c["example"] for c in corrections]

    replay_pool = load_replay(replay_path)
    n_replay = min(len(personal), len(replay_pool))  # 1:1 mix
    replay = random.Random(SEED).sample(replay_pool, n_replay)

    train, valid = [], []
    for ex in personal + replay:
        (valid if _is_valid_split(ex["messages"][1]["content"]) else train).append(ex)
    random.Random(SEED).shuffle(train)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, chunk in (("train", train), ("valid", valid)):
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for ex in chunk:
                f.write(json.dumps(ex) + "\n")

    sha = hashlib.sha256()
    sha.update((out_dir / "train.jsonl").read_bytes())
    sha.update((out_dir / "valid.jsonl").read_bytes())
    return {
        "n_personal": len(personal),
        "n_replay": n_replay,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_banked_negative": len(selection["banked_negative"]),
        "n_deferred": len(selection["deferred"]),
        "n_corrections": len(correction_ids),
        "dataset_sha256": sha.hexdigest(),
        # Id lists for the artifact's provenance manifest. Filtered out of
        # config.json and the dataset.built event, which want counts only.
        "included_ids": selection["included_ids"],
        "banked_negative": selection["banked_negative"],
        "deferred": selection["deferred"],
        "correction_ids": correction_ids,
    }


_SELECTION_EVENTS = {
    "included_ids": "dataset.included",
    "banked_negative": "dataset.banked_negative",
    "deferred": "dataset.deferred",
}


def _emit_selection(store: ResearchStore, selection: dict) -> None:
    """Emit per-exchange selection events on state transition only.

    Arm B retrains from scratch nightly on the whole corpus, so re-emitting
    every membership every night would grow the log without adding information.
    The previous state is read back from the log itself; the full membership of
    any given build lives in that version's provenance manifest.
    """
    previous: dict[str, str] = {}
    for row in store.query(
        "SELECT subject_id, event FROM events WHERE arm = 'lora'"
        " AND event IN ('dataset.included','dataset.banked_negative','dataset.deferred')"
        " ORDER BY id"
    ):
        previous[row["subject_id"]] = row["event"]

    for key, event in _SELECTION_EVENTS.items():
        changed = [eid for eid in selection[key] if previous.get(str(eid)) != event]
        if changed:
            store.emit_all(event, subject_type="exchange", subject_ids=changed,
                           arm="lora")
