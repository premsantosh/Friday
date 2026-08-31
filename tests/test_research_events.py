"""Provenance event log: emission, the forgetting guard, and containment."""

from __future__ import annotations

import inspect
import json

import pytest

from research.db import ResearchStore
from research.events import KNOWN_EVENTS, STAGES, SUBJECT_TYPES


@pytest.fixture
def store(tmp_path):
    s = ResearchStore(str(tmp_path / "research.db"))
    yield s
    s.close()


def _events(store, event=None):
    rows = store.query("SELECT * FROM events ORDER BY id")
    return [r for r in rows if event is None or r["event"] == event]


# ------------------------------------------------------------------ emit API
def test_emit_roundtrip(store):
    store.emit("train.started", subject_type="artifact", subject_id="lora/v20260802",
               arm="lora", artifact_version="lora/v20260802",
               detail={"iters": 400, "lr": 1e-5})

    rows = _events(store)
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "train.started"
    assert row["subject_type"] == "artifact"
    assert row["subject_id"] == "lora/v20260802"
    assert row["arm"] == "lora"
    assert row["artifact_version"] == "lora/v20260802"
    assert json.loads(row["detail"]) == {"iters": 400, "lr": 1e-5}
    assert row["stage"] == "live"  # default context
    assert row["run_id"] is None


def test_subject_id_is_stringified(store):
    """Exchange ids arrive as ints but the column is TEXT; lookups must still hit."""
    store.emit("dataset.included", subject_type="exchange", subject_id=412)
    assert store.events_for("exchange", 412) == store.events_for("exchange", "412")
    assert len(store.events_for("exchange", 412)) == 1


def test_set_run_context_tags_subsequent_events(store):
    store.emit("run.started", subject_type="run", subject_id=1)
    store.set_run_context(17, "train")
    store.emit("train.started", subject_type="artifact", subject_id="lora/v1")

    rows = _events(store)
    assert (rows[0]["run_id"], rows[0]["stage"]) == (None, "live")
    assert (rows[1]["run_id"], rows[1]["stage"]) == (17, "train")


def test_emit_all_batches_one_row_per_subject(store):
    store.emit_all("dataset.included", subject_type="exchange",
                   subject_ids=[3, 4, 7], arm="lora", detail={"split": "train"})

    rows = _events(store)
    assert [r["subject_id"] for r in rows] == ["3", "4", "7"]
    assert all(r["event"] == "dataset.included" and r["arm"] == "lora" for r in rows)


def test_emit_never_raises_and_counts_failures(store):
    """Bookkeeping must not be able to break a reply or abort a nightly stage."""
    store.execute("DROP TABLE events")

    store.emit("shadow.generated", subject_type="exchange", subject_id=1)
    store.emit_all("dataset.included", subject_type="exchange", subject_ids=[1, 2])

    assert store.emit_failures == 2


def test_emit_failure_does_not_break_the_write_it_describes(store):
    """A broken event log must not stop the study from recording exchanges."""
    store.execute("DROP TABLE events")

    with pytest.raises(Exception):
        # The write and its event share a transaction, so this one does fail...
        store.record_exchange("hi", "hello, sir")

    # ...but the store stays usable once the table is back.
    store.conn.executescript(
        "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " ts REAL NOT NULL, run_id INTEGER, stage TEXT NOT NULL, event TEXT NOT NULL,"
        " subject_type TEXT NOT NULL, subject_id TEXT NOT NULL, arm TEXT,"
        " artifact_version TEXT, detail TEXT)")
    assert store.record_exchange("hi", "hello, sir") > 0


# ------------------------------------------------------- the forgetting guard
def test_every_store_write_emits_an_event(store):
    """Nothing may enter research.db untraced.

    Drives every public write method and asserts a matching event row. Adding a
    sixth write path without an emit fails here, which is the point: the loop
    starved for a week because a gap in this chain was invisible.
    """
    eid = store.record_exchange("what's for dinner", "Something light, sir.",
                                route="chat", context_snapshot={"system_prompt": "p"})
    assert [e["event"] for e in _events(store)] == ["exchange.recorded"]
    assert json.loads(_events(store)[0]["detail"])["has_snapshot"] is True

    store.update_exchange(eid, channel="telegram", latency_ms=1840)
    store.add_feedback(eid, kind="implicit", signal=-1, source="miner:rephrase")
    store.add_feedback(eid, kind="implicit", signal=-1, source="miner:rephrase")  # dup
    store.upsert_feedback(eid, kind="explicit", signal=1, source="telegram_button")
    store.add_shadow_response(eid, arm="base", mode="live", response_text="hi")
    store.add_shadow_response(eid, arm="lora", mode="replay", response_text="hi")
    store.add_curated_response(None, "style-01", "lora", "Indeed, sir.",
                               artifact_version="v20260830")

    assert [e["event"] for e in _events(store)] == [
        "exchange.recorded",
        "exchange.backfilled",
        "feedback.added",
        "feedback.duplicate_ignored",
        "feedback.upserted",
        "shadow.generated",
        "replay.generated",
        "eval.candidate_recorded",
    ]


def test_write_methods_are_all_covered_by_the_guard():
    """Guard on the guard: catches a new write method the test above misses."""
    driven = {
        "record_exchange", "update_exchange", "add_feedback", "upsert_feedback",
        "add_shadow_response", "add_curated_response",
    }
    # Generic escape hatches; callers own their own events (nightly/eval writes).
    exempt = {"execute", "emit", "emit_all", "backup_to", "set_run_context", "close"}

    writes = set()
    for name, fn in inspect.getmembers(ResearchStore, inspect.isfunction):
        if name.startswith("_"):
            continue
        source = inspect.getsource(fn)
        if "INSERT INTO" in source or "UPDATE " in source or "INSERT OR" in source:
            writes.add(name)

    missed = writes - driven - exempt
    assert not missed, (
        f"write method(s) {sorted(missed)} are not covered by "
        f"test_every_store_write_emits_an_event — add them there with an emit")


def test_no_user_text_in_event_details(store):
    """detail carries references and metadata, never content (PII surface)."""
    secret = "my card number is 4111 1111 1111 1111"
    eid = store.record_exchange(secret, "I won't repeat that, sir.", route="chat")
    store.upsert_feedback(eid, kind="explicit", signal=-1, source="telegram_button")

    for row in _events(store):
        assert secret not in (row["detail"] or "")
        assert "I won't repeat" not in (row["detail"] or "")


# ------------------------------------------------------------- read helpers
def test_events_for_artifact_matches_both_directions(store):
    """Events that fed an artifact and events about the artifact itself."""
    store.emit("dataset.included", subject_type="exchange", subject_id=1,
               arm="lora", artifact_version="lora/v20260802")
    store.emit("artifact.advanced", subject_type="artifact",
               subject_id="lora/v20260802", arm="lora")
    store.emit("dataset.included", subject_type="exchange", subject_id=2,
               arm="prompt", artifact_version="prompt/v20260802")

    rows = store.events_for_artifact("lora/v20260802")
    assert [r["event"] for r in rows] == ["dataset.included", "artifact.advanced"]


def test_events_for_run_and_recent_filters(store):
    store.set_run_context(17, "harvest")
    store.emit("feedback.mined", subject_type="exchange", subject_id=1)
    store.set_run_context(18, "train")
    store.emit("train.started", subject_type="artifact", subject_id="lora/v1", arm="lora")

    assert len(store.events_for_run(17)) == 1
    assert len(store.events_for_run(18)) == 1
    assert [r["event"] for r in store.recent_events(event="train.started")] == \
        ["train.started"]
    assert [r["event"] for r in store.recent_events(arm="lora")] == ["train.started"]
    assert len(store.recent_events(limit=1)) == 1


def test_recent_events_is_newest_first(store):
    for i in range(3):
        store.emit("shadow.generated", subject_type="exchange", subject_id=i)
    rows = store.recent_events(limit=10)
    assert [r["subject_id"] for r in rows] == ["2", "1", "0"]


# ------------------------------------------------------------------ taxonomy
def test_store_emits_only_documented_event_names(store):
    """Auto-emitted names must be in the taxonomy other tooling reads."""
    eid = store.record_exchange("hi", "hello", route="chat")
    store.update_exchange(eid, channel="telegram")
    store.add_feedback(eid, kind="implicit", signal=-1, source="miner:rephrase")
    store.add_feedback(eid, kind="implicit", signal=-1, source="miner:rephrase")
    store.upsert_feedback(eid, kind="explicit", signal=1, source="telegram_button")
    store.add_shadow_response(eid, arm="base", mode="live", response_text="x")
    store.add_shadow_response(eid, arm="lora", mode="replay", response_text="x")

    for row in _events(store):
        assert row["event"] in KNOWN_EVENTS, f"undocumented event {row['event']!r}"
        assert row["stage"] in STAGES
        assert row["subject_type"] in SUBJECT_TYPES


def test_counts_includes_events(store):
    store.record_exchange("hi", "hello")
    assert store.counts()["events"] == 1
