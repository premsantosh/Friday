"""Tests for arm B's data pipeline and training orchestration logic.

Hermetic: tmp stores, fake correction LLM, monkeypatched training/generation —
no model weights, no subprocesses, no API calls.
"""

from __future__ import annotations

import json
import time

import pytest

from research import artifacts
from research.approaches import lora_pipeline, train_lora
from research.db import ResearchStore
from research.persona import PERSONA_PROMPT


@pytest.fixture
def store(tmp_path):
    s = ResearchStore(str(tmp_path / "research.db"))
    yield s
    s.close()


NOW = time.time()
OLD = NOW - 3 * 24 * 3600  # comfortably past the 48h neutral window


def _chat(store, user, reply, ts=OLD, uid="u"):
    return store.record_exchange(user, reply, route="chat", user_id=uid, ts=ts)


# ------------------------------------------------------------------ selection

def test_selection_partitions_by_feedback_and_age(store):
    liked = _chat(store, "liked message", "good reply")
    store.add_feedback(liked, kind="explicit", signal=1, source="telegram_button")
    disliked = _chat(store, "disliked message", "bad reply")
    store.add_feedback(disliked, kind="explicit", signal=-1, source="telegram_button")
    _chat(store, "old neutral message", "fine reply")                # included
    fresh = _chat(store, "fresh neutral message", "reply", ts=NOW)   # deferred
    store.record_exchange("lights on", "Done.", route="router:hue", ts=OLD)  # not chat

    sel = lora_pipeline.select_personal_examples(store, now=NOW)
    included_users = [ex["messages"][1]["content"] for ex in sel["included"]]
    assert included_users == ["liked message", "old neutral message"]
    assert sel["banked_negative"] == [disliked]
    assert sel["deferred"] == [fresh]


def test_liked_but_fresh_is_included_immediately(store):
    liked = _chat(store, "fresh but liked", "reply", ts=NOW)
    store.add_feedback(liked, kind="explicit", signal=1, source="telegram_button")
    sel = lora_pipeline.select_personal_examples(store, now=NOW)
    assert len(sel["included"]) == 1


def test_example_uses_pinned_persona_prompt(store):
    _chat(store, "hello", "Good evening, sir.")
    sel = lora_pipeline.select_personal_examples(store, now=NOW)
    assert sel["included"][0]["messages"][0] == {"role": "system",
                                                "content": PERSONA_PROMPT}


# ---------------------------------------------------------------- corrections

def test_correction_synthesis_cached_once(store, tmp_path):
    eid = _chat(store, "remind me about the birthday", "It is on Monday, sir.")
    _chat(store, "no, it's on Friday", "Apologies, sir.")
    store.add_feedback(eid, kind="implicit", signal=-1, source="miner:correction")

    calls = []

    def llm(prompt):
        calls.append(prompt)
        return "It is on Friday, sir."

    cache = tmp_path / "corrections.jsonl"
    examples = lora_pipeline.synthesize_corrections(store, cache, llm)
    assert len(examples) == 1
    assert examples[0]["exchange_id"] == eid
    assert examples[0]["example"]["messages"][2]["content"] == "It is on Friday, sir."
    assert "no, it's on Friday" in calls[0]

    # Second build: served from cache, no second paid call.
    examples2 = lora_pipeline.synthesize_corrections(store, cache, llm)
    assert len(examples2) == 1
    assert len(calls) == 1


def test_correction_synthesis_uses_followup_id_from_details(store, tmp_path):
    """The miner recorded which message corrected; don't re-derive it.

    Regression: the old lookup matched `user_id IS ?`, so with a NULL user_id
    (record_chat inserts NULL) it picked the next exchange by any user, which on
    a busy stretch is the wrong sentence entirely.
    """
    eid = _chat(store, "remind me about the birthday", "It is on Monday, sir.")
    _chat(store, "unrelated question", "Indeed, sir.")
    correcting = _chat(store, "no, it's on Friday", "Apologies, sir.")
    store.add_feedback(eid, kind="implicit", signal=-1, source="miner:correction",
                       details=json.dumps({"followup_id": correcting,
                                           "opener": "no,"}))

    calls = []
    lora_pipeline.synthesize_corrections(store, tmp_path / "c.jsonl",
                                         lambda p: (calls.append(p), "Friday, sir.")[1])

    assert "no, it's on Friday" in calls[0]
    assert "unrelated question" not in calls[0]


def test_correction_synthesis_tolerates_legacy_details(store, tmp_path):
    """Rows written before details became JSON must not break the build."""
    eid = _chat(store, "remind me about the birthday", "It is on Monday, sir.")
    _chat(store, "no, it's on Friday", "Apologies, sir.")
    store.add_feedback(eid, kind="implicit", signal=-1, source="miner:correction",
                       details="opener='no,'")  # old free-text format

    calls = []
    examples = lora_pipeline.synthesize_corrections(
        store, tmp_path / "c.jsonl", lambda p: (calls.append(p), "Friday, sir.")[1])

    assert len(examples) == 1
    assert "no, it's on Friday" in calls[0]  # falls back to the next exchange


# -------------------------------------------------------------------- dataset

def test_build_dataset_mixes_replay_1to1_and_splits(store, tmp_path):
    for i in range(10):
        _chat(store, f"personal message number {i}", f"reply {i}")
    out = tmp_path / "data"
    stats = lora_pipeline.build_dataset(store, out, now=NOW)

    assert stats["n_personal"] == 10
    assert stats["n_replay"] == 10  # 1:1 mix
    assert stats["n_train"] + stats["n_valid"] == 20
    assert stats["n_valid"] >= 1
    assert len(stats["dataset_sha256"]) == 64

    train_lines = (out / "train.jsonl").read_text().splitlines()
    ex = json.loads(train_lines[0])
    assert ex["messages"][0]["role"] == "system"

    # Deterministic: same inputs, same hash.
    stats2 = lora_pipeline.build_dataset(store, tmp_path / "data2", now=NOW)
    assert stats2["dataset_sha256"] == stats["dataset_sha256"]


def test_valid_split_is_stable_per_text():
    text = "some user message"
    assert lora_pipeline._is_valid_split(text) == lora_pipeline._is_valid_split(text)


# ------------------------------------------------------------ train_nightly

def test_train_skips_below_min_examples(store, tmp_path, monkeypatch):
    monkeypatch.setattr(train_lora, "free_memory_pct", lambda: 50)
    monkeypatch.setattr(train_lora, "unload_ollama", lambda *a, **k: None)
    _chat(store, "only one message", "reply")
    note = train_lora.train_nightly(store, "20260718", artifacts_dir=tmp_path)
    assert "skipped: 1 personal examples" in note
    assert artifacts.current_version("lora", tmp_path) is None


def test_train_skips_on_low_memory(store, tmp_path, monkeypatch):
    monkeypatch.setattr(train_lora, "free_memory_pct", lambda: 5)
    monkeypatch.setattr(train_lora, "unload_ollama", lambda *a, **k: None)
    monkeypatch.setattr(train_lora.time, "sleep", lambda s: None)
    note = train_lora.train_nightly(store, "20260718", artifacts_dir=tmp_path)
    assert note.startswith("skipped: only 5%")


def _seed_trainable(store, n=30):
    for i in range(n):
        _chat(store, f"unique personal message {i} about topic {i}", f"reply {i}")


def test_train_gate_failure_keeps_current(store, tmp_path, monkeypatch):
    _seed_trainable(store)
    monkeypatch.setattr(train_lora, "free_memory_pct", lambda: 50)
    monkeypatch.setattr(train_lora, "unload_ollama", lambda *a, **k: None)
    monkeypatch.setattr(train_lora, "run_training", lambda *a, **k: 0)
    monkeypatch.setattr(train_lora, "quality_gate",
                        lambda p: (False, "style collapse: 0.20 < 0.6"))

    note = train_lora.train_nightly(store, "20260718", artifacts_dir=tmp_path)
    assert "GATED" in note
    assert artifacts.current_version("lora", tmp_path) is None
    vdir = artifacts.arm_dir("lora", tmp_path) / "v20260718"
    assert (vdir / "GATED").exists()
    assert json.loads((vdir / "config.json").read_text())["seed"] == 42


def test_train_pass_advances_current(store, tmp_path, monkeypatch):
    _seed_trainable(store)
    monkeypatch.setattr(train_lora, "free_memory_pct", lambda: 50)
    monkeypatch.setattr(train_lora, "unload_ollama", lambda *a, **k: None)
    monkeypatch.setattr(train_lora, "run_training", lambda *a, **k: 0)
    monkeypatch.setattr(train_lora, "quality_gate", lambda p: (True, "style 0.90"))

    note = train_lora.train_nightly(store, "20260718", artifacts_dir=tmp_path)
    assert note.startswith("advanced to v20260718")
    assert artifacts.current_version("lora", tmp_path) == "v20260718"


def test_train_failure_reported(store, tmp_path, monkeypatch):
    _seed_trainable(store)
    monkeypatch.setattr(train_lora, "free_memory_pct", lambda: 50)
    monkeypatch.setattr(train_lora, "unload_ollama", lambda *a, **k: None)
    monkeypatch.setattr(train_lora, "run_training", lambda *a, **k: 1)
    note = train_lora.train_nightly(store, "20260718", artifacts_dir=tmp_path)
    assert note.startswith("FAILED: mlx_lm lora exit 1")
    assert artifacts.current_version("lora", tmp_path) is None
