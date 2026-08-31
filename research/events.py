"""The provenance event taxonomy: every signal the loop uses to improve itself.

The `events` table in research.db is an append-only timeline answering two
questions the study could not answer before:

    what happened to exchange 412?      -> trace --exchange 412
    what did lora/v20260802 consume?    -> trace --artifact lora/v20260802

Two design rules make that work:

1. The *subject* of an event is the thing you would want to trace, not the row
   that was written. A feedback insert has subject_type='exchange' with the
   feedback row id in `detail`, so one indexed lookup returns a turn's whole
   life story rather than fragments scattered across subject types.

2. `detail` never holds user text or response text. Both already live in
   `exchanges` / `shadow_responses`; duplicating them doubles the PII surface
   for no gain. `detail` carries ids, counts, durations, reasons, model tags
   and gate outcomes, and the reader joins for content.

Emission is wired two ways. Every write path on ResearchStore emits inside its
own transaction, so nothing can enter the database untraced (enforced by
tests/test_research_events.py::test_every_store_write_emits_an_event). Decision
events cannot be inferred from a write, so they are emitted at their site; the
names below are the authoritative list and a test asserts a full nightly
--dry-run emits nothing outside it.

Volume: arm B retrains from scratch nightly on the whole corpus, so emitting a
per-exchange dataset event every night would re-emit the entire history forever.
Per-exchange dataset events therefore fire on STATE TRANSITION ONLY (first
inclusion, first banking, deferred -> included); `dataset.built` carries the
nightly counts and the manifest holds the full membership. That keeps the log at
roughly 150k rows and 40 MB a year, which sqlite does not notice.
"""

from __future__ import annotations

# Live path (stage='live', run_id NULL): the assistant and shadow thread.
LIVE = (
    "exchange.recorded",          # a turn entered the study
    "exchange.backfilled",        # routing metadata added to an existing row
    "feedback.added",             # first-wins insert (miners)
    "feedback.upserted",          # latest-wins upsert (Telegram buttons)
    "feedback.duplicate_ignored", # first-wins insert lost the race, by design
    "shadow.enqueued",
    "shadow.dropped",             # queue full; the reply already went out
    "shadow.generated",
    "shadow.failed",              # the one that was silently swallowed before
)

HARVEST = (
    "feedback.mined",             # a miner produced a signal
    "db.backed_up",
)

REFLECT = (                       # arm A
    "memory.consumed",            # exchange fed to the observer
    "memory.observed",
    "memory.reflected",
)

EVOLVE = (                        # arm C
    "prompt.consumed",            # exchange fed to the evolver
    "prompt.rejected",            # unparseable output or over the size cap
)

TRAIN = (                         # arm B
    "dataset.included",
    "dataset.banked_negative",    # held back from SFT for phase-2 preference tuning
    "dataset.deferred",           # unflagged and too fresh to judge
    "correction.synthesized",
    "correction.cache_hit",
    "dataset.built",
    "train.started",
    "train.finished",
    "gate.passed",
    "gate.failed",
)

REPLAY = (
    "replay.generated",
    "replay.failed",
    "replay.context",             # hash of an unversioned arm's injected block
)

EVAL = (
    "judge.verdict",
    "eval.summary",
    "protocol.evaluated",
    "eval.candidate_recorded",    # a curated-split generation persisted
)

REPORT = ("report.written",)

PUBLISH = (
    "results.committed",          # aggregates committed to git (never pushed)
    "results.blocked",            # PII gate refused the commit
)

# Emitted by any stage that produces a versioned artifact.
ARTIFACT = (
    "artifact.created",
    "artifact.advanced",          # `current` pointer moved
    "artifact.gated",             # built but not promoted (forgetting canary)
)

RUN = (
    "run.started",
    "run.stage_ok",
    "run.stage_failed",
    "run.finished",
)

KNOWN_EVENTS = frozenset(
    LIVE + HARVEST + REFLECT + EVOLVE + TRAIN + REPLAY + EVAL + REPORT
    + PUBLISH + ARTIFACT + RUN
)

# Valid `stage` values. 'live' is the running assistant; the rest are nightly.
STAGES = frozenset({
    "live", "harvest", "reflect", "evolve", "train", "replay", "eval",
    "report", "publish", "run",
})

SUBJECT_TYPES = frozenset({
    "exchange", "memory", "artifact", "dataset", "run", "prompt",
})
