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

# Marks an annotation only the user can write (see curated.yaml's header).
PLACEHOLDER_MARKER = "FILL-IN"

VALID_CATEGORIES = {
    "preference_recall",      # does it remember what the user likes
    "style",                  # persona compliance (brevity, tone, address)
    "routine",                # knowledge of the user's habits/schedule
    "correction_persistence", # does a past correction stick
    "generic_control",        # no personalization needed — regression canary
    "harvested",              # a real prompt the user sent; category unknown
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


def count_placeholders(path: Path = CURATED_PATH) -> int:
    """How many NON-DRAFT curated probes still carry a FILL-IN annotation.

    Annotations are injected verbatim into the judge rubric as known facts about
    the user, so a placeholder does not merely fail to help — it tells the judge
    something false. Surfaced by `research status` until they're written.
    """
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return 0
    return sum(1 for item in raw.get("prompts", [])
               if PLACEHOLDER_MARKER in str(item.get("annotations", ""))
               and not item.get("draft"))


def count_drafts(path: Path = CURATED_PATH) -> int:
    """Skeleton probes (`draft: true`) awaiting the user's annotations.

    Drafts never load into any split, so they cannot poison a verdict — but the
    curated set stays undersized until they're written. Surfaced by `research
    status` and the weekly digest.
    """
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return 0
    return sum(1 for item in raw.get("prompts", []) if item.get("draft"))


def load_curated(path: Path = CURATED_PATH, *,
                 allow_placeholders: bool = False,
                 include_reserve: bool = False,
                 reserve_only: bool = False) -> list[EvalPrompt]:
    """Curated probes, refusing placeholder annotations by default.

    Annotations go verbatim into the judge rubric as "known facts about this
    user" (research/judge.py). A FILL-IN placeholder therefore does not merely
    fail to help — it tells the judge that a known fact about the user is the
    string "FILL-IN: your actual coffee order", and every verdict on that probe
    is noise dressed as a measurement. Refusing beats warning.

    Probe flags:
      draft: true    — a skeleton the user hasn't annotated yet. Excluded from
                       every load and exempt from the FILL-IN refusal, so
                       adding skeletons can never kill the weekly eval.
      reserve: true  — held-out probes for the final paper numbers only.
                       Excluded unless include_reserve/reserve_only is set
                       (`research eval --split reserve`).
    """
    raw = yaml.safe_load(path.read_text())
    prompts = []
    seen_ids = set()
    placeholders = []
    for item in raw["prompts"]:
        pid = str(item["id"])
        if pid in seen_ids:
            raise ValueError(f"duplicate eval prompt id: {pid}")
        seen_ids.add(pid)
        if item.get("draft"):
            continue
        is_reserve = bool(item.get("reserve"))
        if reserve_only and not is_reserve:
            continue
        if is_reserve and not (include_reserve or reserve_only):
            continue
        category = item["category"]
        if category not in VALID_CATEGORIES:
            raise ValueError(f"unknown category {category!r} for prompt {pid}")
        annotations = item.get("annotations", "")
        if PLACEHOLDER_MARKER in str(annotations):
            placeholders.append(pid)
        prompts.append(EvalPrompt(
            id=pid,
            prompt=item["prompt"],
            category=category,
            annotations=annotations,
            meta={"reserve": is_reserve},
        ))
    if placeholders and not allow_placeholders:
        raise ValueError(
            f"{len(placeholders)} curated probe(s) still have {PLACEHOLDER_MARKER} "
            f"annotations: {', '.join(placeholders)}. Fill them in "
            f"({path}) or pass allow_placeholders=True for a smoke run.")
    return prompts


def load_harvested(store: ResearchStore, *, after_ts: float,
                   limit: int = 50) -> list[EvalPrompt]:
    """Real chat prompts newer than `after_ts` (the training-data cutoff).

    Only route='chat' exchanges qualify — workflow commands ("lights on") are
    not free-conversation and their replies aren't comparable across arms.
    """
    rows = store.query(
        "SELECT id, ts, user_text FROM exchanges"
        " WHERE route = 'chat' AND ts > ? ORDER BY ts LIMIT ?",
        (after_ts, limit),
    )
    return [
        EvalPrompt(
            id=f"harvest-{r['id']}",
            prompt=r["user_text"],
            # Whatever the user happened to say. Labelling it a personalization
            # probe would put a fiction into the results.
            category="harvested",
            source="harvested",
            ts=r["ts"],
            meta={"exchange_id": r["id"]},
        )
        for r in rows
    ]
