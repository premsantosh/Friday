"""Eval set loading and construction.

Two sources of eval prompts:
  * Curated probes (research/data/evalset/curated.yaml): hand-written prompts
    with annotations of the preferences/style a good answer must honor.
    PII-reviewed before committing — annotations describe preferences, never
    secrets.
  * Harvested prompts: real chat exchanges pulled from research.db with a
    temporal split — an artifact trained on data through day N is only ever
    evaluated on prompts from day N+1 onward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from research.db import ResearchStore

CURATED_PATH = Path(__file__).parent / "data" / "evalset" / "curated.yaml"

VALID_CATEGORIES = {
    "preference_recall",      # does it remember what the user likes
    "style",                  # persona compliance (brevity, tone, address)
    "routine",                # knowledge of the user's habits/schedule
    "correction_persistence", # does a past correction stick
    "generic_control",        # no personalization needed — regression canary
}


@dataclass
class EvalPrompt:
    id: str
    prompt: str
    category: str
    annotations: str = ""     # known preferences/constraints a good answer honors
    source: str = "curated"   # curated | harvested
    ts: float = 0.0           # harvest time (temporal splits); 0 for curated
    meta: dict = field(default_factory=dict)


def load_curated(path: Path = CURATED_PATH) -> list[EvalPrompt]:
    raw = yaml.safe_load(path.read_text())
    prompts = []
    seen_ids = set()
    for item in raw["prompts"]:
        pid = str(item["id"])
        if pid in seen_ids:
            raise ValueError(f"duplicate eval prompt id: {pid}")
        seen_ids.add(pid)
        category = item["category"]
        if category not in VALID_CATEGORIES:
            raise ValueError(f"unknown category {category!r} for prompt {pid}")
        prompts.append(EvalPrompt(
            id=pid,
            prompt=item["prompt"],
            category=category,
            annotations=item.get("annotations", ""),
        ))
    return prompts


def load_harvested(store: ResearchStore, *, after_ts: float,
                   limit: int = 50) -> list[EvalPrompt]:
    """Real chat prompts newer than `after_ts` (the training-data cutoff).

    Only route='chat' exchanges qualify — workflow commands ("lights on") are
    not free-conversation and their replies aren't comparable across arms.
    """
    rows = store.conn.execute(
        "SELECT id, ts, user_text FROM exchanges"
        " WHERE route = 'chat' AND ts > ? ORDER BY ts LIMIT ?",
        (after_ts, limit),
    ).fetchall()
    return [
        EvalPrompt(
            id=f"harvest-{r['id']}",
            prompt=r["user_text"],
            category="preference_recall",
            source="harvested",
            ts=r["ts"],
            meta={"exchange_id": r["id"]},
        )
        for r in rows
    ]
