"""Tests for the research store (research/db.py). Hermetic: tmp_path DB only."""

from __future__ import annotations

import threading

import pytest

from research.db import ResearchStore


@pytest.fixture
def store(tmp_path):
    s = ResearchStore(str(tmp_path / "research.db"))
    yield s
    s.close()


def test_record_and_get_exchange_roundtrip(store):
    eid = store.record_exchange(
        "what's the weather", "Rainy, sir.",
        route="chat", channel="telegram", user_id="555",
        latency_ms=1200, model="Anthropic (haiku)",
        context_snapshot={"system_prompt": "You are Jarvis", "messages": [{"role": "user", "content": "hi"}]},
        memory_turn_id=7,
    )
    row = store.get_exchange(eid)
    assert row["user_text"] == "what's the weather"
    assert row["reply_text"] == "Rainy, sir."
    assert row["route"] == "chat"
    assert row["context_snapshot"]["system_prompt"] == "You are Jarvis"
    assert row["memory_turn_id"] == 7
    assert row["ts"] > 0


def test_update_exchange_backfills_metadata_only(store):
    eid = store.record_exchange("hi", "Hello, sir.", route="chat")
    store.update_exchange(eid, channel="telegram", user_id="555", latency_ms=900)
    row = store.get_exchange(eid)
    assert (row["channel"], row["user_id"], row["latency_ms"]) == ("telegram", "555", 900)

    # Content is append-only — never updatable.
    with pytest.raises(ValueError):
        store.update_exchange(eid, reply_text="rewritten")


def test_update_exchange_ignores_none_values(store):
    eid = store.record_exchange("hi", "Hello.", route="chat", channel="local")
    store.update_exchange(eid, channel=None, latency_ms=5)
    row = store.get_exchange(eid)
    assert row["channel"] == "local"
    assert row["latency_ms"] == 5


def test_feedback_roundtrip(store):
    eid = store.record_exchange("hi", "Hello.", route="chat")
    store.add_feedback(eid, kind="explicit", signal=1, source="telegram_button")
    store.add_feedback(eid, kind="implicit", signal=-1, source="miner:rephrase", details="similarity=0.8")
    fb = store.feedback_for(eid)
    assert [(f["kind"], f["signal"]) for f in fb] == [("explicit", 1), ("implicit", -1)]
    assert store.has_feedback(eid, "miner:rephrase")
    assert not store.has_feedback(eid, "miner:correction")


def test_shadow_response_roundtrip(store):
    eid = store.record_exchange("hi", "Hello.", route="chat")
    store.add_shadow_response(eid, arm="base", mode="live", response_text="Greetings.",
                              model_tag="llama3.1", gen_ms=800)
    assert store.counts()["shadow_responses"] == 1


def test_exchanges_since_orders_by_time(store):
    store.record_exchange("first", "a", route="chat", ts=100.0)
    store.record_exchange("second", "b", route="chat", ts=200.0)
    store.record_exchange("old", "c", route="chat", ts=50.0)
    since = store.exchanges_since(90.0)
    assert [e["user_text"] for e in since] == ["first", "second"]


def test_concurrent_writes_do_not_corrupt(store):
    def write(n):
        for i in range(20):
            store.record_exchange(f"msg-{n}-{i}", "r", route="chat")

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.counts()["exchanges"] == 80
