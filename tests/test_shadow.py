"""Tests for the shadow runner (research/shadow.py). Hermetic via injected poster."""

from __future__ import annotations

import time

import pytest

from research.db import ResearchStore
from research.shadow import ShadowRunner


class FakeResp:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload or {"message": {"content": "shadow says hi"}}
        self.status_code = status_code

    def json(self):
        return self._payload


@pytest.fixture
def store(tmp_path):
    s = ResearchStore(str(tmp_path / "research.db"))
    yield s
    s.close()


def _chat_exchange(store):
    return store.record_exchange(
        "what's for dinner", "Pasta, sir.", route="chat",
        context_snapshot={"system_prompt": "You are Jarvis",
                          "messages": [{"role": "user", "content": "what's for dinner"}]},
    )


def test_process_posts_snapshot_and_stores_response(store):
    posted = {}

    def poster(url, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        return FakeResp()

    runner = ShadowRunner(store, model_tag="llama3.1", poster=poster)
    eid = _chat_exchange(store)
    runner._process(eid)

    assert posted["url"].endswith("/api/chat")
    assert posted["json"]["model"] == "llama3.1"
    assert posted["json"]["messages"][0] == {"role": "system", "content": "You are Jarvis"}
    assert posted["json"]["messages"][-1]["content"] == "what's for dinner"
    assert "think" not in posted["json"]  # llama doesn't accept the think option

    rows = store.conn.execute("SELECT arm, mode, response_text FROM shadow_responses").fetchall()
    assert [tuple(r) for r in rows] == [("base", "live", "shadow says hi")]


def test_thinking_model_gets_think_false_and_output_stripped(store):
    posted = {}

    def poster(url, json=None, timeout=None):
        posted["json"] = json
        return FakeResp({"message": {"content":
            "<think>the user wants dinner ideas</think>Pasta would suit, sir."}})

    runner = ShadowRunner(store, model_tag="qwen3:8b", poster=poster)
    runner._process(_chat_exchange(store))

    assert posted["json"]["think"] is False
    rows = store.conn.execute("SELECT response_text FROM shadow_responses").fetchall()
    assert [r["response_text"] for r in rows] == ["Pasta would suit, sir."]


def test_strip_think_variants():
    from research.generate import strip_think

    assert strip_think("<think>hmm</think>Answer.") == "Answer."
    assert strip_think("Answer.<think>trailing unclosed") == "Answer."
    assert strip_think("Plain answer.") == "Plain answer."
    assert strip_think("<think>a</think>Mid<think>b</think> end") == "Mid end"


def test_exchange_without_snapshot_is_skipped(store):
    def poster(url, json=None, timeout=None):  # pragma: no cover - must not be called
        raise AssertionError("should not post without a snapshot")

    runner = ShadowRunner(store, poster=poster)
    eid = store.record_exchange("hi", "Hello.", route="keyword:time")
    runner._process(eid)
    assert store.counts()["shadow_responses"] == 0


def test_worker_swallows_poster_exceptions(store):
    def poster(url, json=None, timeout=None):
        raise ConnectionError("ollama down")

    runner = ShadowRunner(store, poster=poster)
    runner.start()
    try:
        eid = _chat_exchange(store)
        assert runner.enqueue(eid) is True
        deadline = time.monotonic() + 2
        while runner._queue.qsize() and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        runner.stop()
    # No exception escaped, nothing stored.
    assert store.counts()["shadow_responses"] == 0


def test_enqueue_drops_when_queue_full(store):
    runner = ShadowRunner(store, poster=lambda *a, **k: FakeResp(), max_queue=2)
    # Worker not started — the queue fills and overflow drops.
    assert runner.enqueue(1) is True
    assert runner.enqueue(2) is True
    assert runner.enqueue(3) is False


def test_worker_end_to_end_via_queue(store):
    runner = ShadowRunner(store, poster=lambda url, json=None, timeout=None: FakeResp())
    runner.start()
    try:
        eid = _chat_exchange(store)
        runner.enqueue(eid)
        deadline = time.monotonic() + 2
        while store.counts()["shadow_responses"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        runner.stop()
    assert store.counts()["shadow_responses"] == 1
