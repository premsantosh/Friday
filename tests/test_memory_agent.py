"""Tests for arm A (research/approaches/memory_agent.py).

Hermetic: fake LLM functions and a fake vector index; no Ollama, no ChromaDB.
"""

from __future__ import annotations

import json
import time

import pytest

from research import artifacts
from research.approaches.memory_agent import REFLECT_EVERY, MemoryAgent, _parse_items
from research.db import ResearchStore


@pytest.fixture
def store(tmp_path):
    s = ResearchStore(str(tmp_path / "research.db"))
    yield s
    s.close()


@pytest.fixture
def art_dir(tmp_path):
    return tmp_path / "artifacts"


class FakeIndex:
    def __init__(self):
        self.docs: dict[int, str] = {}
        self.similarities: dict[int, float] = {}

    def add(self, memory_id, text):
        self.docs[memory_id] = text

    def query(self, text, k):
        return sorted(self.similarities.items(), key=lambda kv: -kv[1])[:k]


def _seed_chat(store):
    store.record_exchange("I always take oat milk", "Noted, sir.", route="chat",
                          ts=time.time())


# --------------------------------------------------------------------- parsing

def test_parse_items_tolerates_prose_and_clamps():
    raw = 'Here you go: [{"text": "prefers oat milk", "importance": 99},' \
          ' {"text": "", "importance": 3}, {"text": "runs at dawn", "importance": "high"}]'
    items = _parse_items(raw)
    assert items == [{"text": "prefers oat milk", "importance": 10.0},
                     {"text": "runs at dawn", "importance": 5.0}]


def test_parse_items_garbage_is_empty():
    assert _parse_items("no json") == []


# --------------------------------------------------------------------- observe

def test_observe_inserts_memories_and_indexes(store, art_dir):
    _seed_chat(store)
    index = FakeIndex()
    agent = MemoryAgent(
        store, artifacts_dir=art_dir, index=index,
        llm_fn=lambda p: json.dumps([{"text": "prefers oat milk", "importance": 8}]),
    )
    assert agent.observe(since_ts=0.0) == 1
    row = store.conn.execute("SELECT kind, text, importance FROM memories").fetchone()
    assert tuple(row) == ("observation", "prefers oat milk", 8.0)
    assert index.docs == {1: "prefers oat milk"}


def test_observe_skips_without_chat(store, art_dir):
    agent = MemoryAgent(store, artifacts_dir=art_dir,
                        llm_fn=lambda p: pytest.fail("must not call LLM"))
    assert agent.observe(since_ts=0.0) == 0


# --------------------------------------------------------------------- reflect

def test_reflect_only_after_threshold(store, art_dir):
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return json.dumps([{"text": "user is a morning person", "importance": 9}])

    agent = MemoryAgent(store, artifacts_dir=art_dir, llm_fn=llm)
    for i in range(REFLECT_EVERY - 1):
        agent._insert([{"text": f"obs {i}", "importance": 5.0}],
                      kind="observation", sources=[])
    assert agent.maybe_reflect() == 0
    assert calls == []

    agent._insert([{"text": "one more", "importance": 5.0}],
                  kind="observation", sources=[])
    assert agent.maybe_reflect() == 1
    assert store.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE kind='reflection'").fetchone()[0] == 1

    # Counter resets after a reflection: no immediate second pass.
    assert agent.maybe_reflect() == 0


# -------------------------------------------------------------------- retrieve

def test_retrieve_ranks_by_similarity_recency_importance(store, art_dir):
    index = FakeIndex()
    agent = MemoryAgent(store, artifacts_dir=art_dir, index=index, llm_fn=lambda p: "[]")
    now = time.time()
    with store._lock:
        store.conn.execute(
            "INSERT INTO memories (ts, kind, text, importance) VALUES (?, 'observation', 'old low', 2)",
            (now - 60 * 86400,))
        store.conn.execute(
            "INSERT INTO memories (ts, kind, text, importance) VALUES (?, 'observation', 'fresh high', 9)",
            (now,))
        store.conn.execute(
            "INSERT INTO memories (ts, kind, text, importance) VALUES (?, 'observation', 'similar', 5)",
            (now - 30 * 86400,))
        store.conn.commit()
    index.similarities = {3: 1.0}  # 'similar' is the only vector match

    top = agent.retrieve("query", k=2)
    assert [m["text"] for m in top] == ["similar", "fresh high"]


def test_retrieve_respects_retired_and_max_id(store, art_dir):
    agent = MemoryAgent(store, artifacts_dir=art_dir, llm_fn=lambda p: "[]")
    now = time.time()
    with store._lock:
        for text in ("first", "second", "third"):
            store.conn.execute(
                "INSERT INTO memories (ts, kind, text, importance) VALUES (?, 'observation', ?, 5)",
                (now, text))
        store.conn.execute("UPDATE memories SET retired = 1 WHERE text = 'second'")
        store.conn.commit()
    assert {m["text"] for m in agent.retrieve("q", k=10)} == {"first", "third"}
    assert {m["text"] for m in agent.retrieve("q", k=10, max_id=2)} == {"first"}


def test_system_block_formats_or_empty(store, art_dir):
    agent = MemoryAgent(store, artifacts_dir=art_dir, llm_fn=lambda p: "[]")
    assert agent.system_block_for("q") == ""
    agent._insert([{"text": "prefers oat milk", "importance": 8.0}],
                  kind="observation", sources=[1])
    block = agent.system_block_for("q")
    assert block.startswith("<learned_memory>")
    assert "- prefers oat milk" in block


# ------------------------------------------------------------------ versioning

def test_snapshot_pins_max_id_and_advances(store, art_dir):
    agent = MemoryAgent(store, artifacts_dir=art_dir, llm_fn=lambda p: "[]")
    agent._insert([{"text": "a", "importance": 5.0}], kind="observation", sources=[])
    v1 = agent.snapshot("20260718")
    agent._insert([{"text": "b", "importance": 5.0}], kind="observation", sources=[])
    v2 = agent.snapshot("20260719")

    assert artifacts.current_version("memory", art_dir) == v2
    assert agent.current_max_id() == 2

    # Revert to v1: retrieval pinned to the first memory only.
    artifacts.advance_current("memory", v1, art_dir)
    assert agent.current_max_id() == 1
    assert {m["text"] for m in agent.retrieve("q", max_id=agent.current_max_id())} == {"a"}


# ------------------------------------------------------------- facts baseline

def test_facts_baseline_block(tmp_path):
    from memory.store import FridayStore
    from research.approaches.facts_baseline import system_block_for

    fstore = FridayStore(str(tmp_path / "memory.db"))
    fstore.remember("coffee_order", "oat milk flat white", category="preference")
    block = system_block_for("make me a coffee order", store=fstore)
    assert "<known_facts>" in block and "oat milk flat white" in block
    assert system_block_for("unrelated query about volcanoes", store=fstore) == ""
