"""Tests for ConversationRecorder and its provider/assistant hooks.

Hermetic: research store on tmp_path; the LLM provider is a minimal fake with
in-memory stand-ins for FridayStore/ContextBuilder, so nothing touches
~/.friday or the network.
"""

from __future__ import annotations

import time

import pytest

from config import LLMConfig, PersonalityConfig
from llm.providers import LLMProvider
from research.db import ResearchStore
from research.recorder import ConversationRecorder


@pytest.fixture
def store(tmp_path):
    s = ResearchStore(str(tmp_path / "research.db"))
    yield s
    s.close()


@pytest.fixture
def recorder(store):
    return ConversationRecorder(store)


# ----------------------------------------------------------------- recorder

def test_record_chat_then_record_turn_backfills_not_duplicates(store, recorder):
    eid = recorder.record_chat("hello", "Good evening, sir.",
                               model="fake", context_snapshot={"system_prompt": "sp", "messages": []})
    turn_id = recorder.record_turn("hello", "Good evening, sir.", route="chat",
                                   latency_ms=1500, user_id="555", channel="telegram")
    assert turn_id == eid
    assert store.counts()["exchanges"] == 1
    row = store.get_exchange(eid)
    assert (row["channel"], row["user_id"], row["latency_ms"]) == ("telegram", "555", 1500)
    assert row["route"] == "chat"


def test_record_turn_alone_inserts_workflow_exchange(store, recorder):
    recorder.record_turn("turn on the lights", "Done, sir.", route="router:philips_hue",
                         latency_ms=50, user_id="555", channel="telegram")
    assert store.counts()["exchanges"] == 1


def test_workflow_failure_banks_an_implicit_negative(store, recorder):
    eid = recorder.record_turn("turn on the lights", "The bridge is offline, sir.",
                               route="router:philips_hue", user_id="555",
                               outcome="failure")
    rows = store.feedback_for(eid)
    assert len(rows) == 1
    assert (rows[0]["kind"], rows[0]["signal"], rows[0]["source"]) == \
        ("implicit", -1, "workflow_failure")


def test_workflow_success_banks_nothing(store, recorder):
    """Successful routing says nothing about reply quality — not a +1."""
    eid = recorder.record_turn("turn on the lights", "Done, sir.",
                               route="router:philips_hue", user_id="555",
                               outcome="success")
    assert store.feedback_for(eid) == []


def test_shadow_enqueued_on_chat_record(store):
    enqueued = []

    class FakeShadow:
        def enqueue(self, eid):
            enqueued.append(eid)

    recorder = ConversationRecorder(store, shadow=FakeShadow())
    eid = recorder.record_chat("hi", "Hello.", context_snapshot={"system_prompt": "s", "messages": []})
    assert enqueued == [eid]


# ----------------------------------------------------------------- feedback

def test_feedback_markup_only_for_fresh_chat_exchange(store, recorder):
    recorder.record_chat("hi", "Hello.")
    recorder.record_turn("hi", "Hello.", route="chat", user_id="555")
    markup = recorder.feedback_markup("555", "Hello.")
    eid = store.counts()["exchanges"]
    assert markup == {"inline_keyboard": [[
        {"text": "👍", "callback_data": f"fb:{eid}:1"},
        {"text": "👎", "callback_data": f"fb:{eid}:0"},
    ]]}


def test_feedback_markup_absent_for_workflow_reply(store, recorder):
    recorder.record_turn("lights on", "Done.", route="router:philips_hue", user_id="555")
    assert recorder.feedback_markup("555", "Done.") is None


def test_feedback_markup_absent_when_stale(store, recorder, monkeypatch):
    recorder.record_chat("hi", "Hello.")
    recorder.record_turn("hi", "Hello.", route="chat", user_id="555")
    real = time.monotonic()
    monkeypatch.setattr("research.recorder.time.monotonic", lambda: real + 120)
    assert recorder.feedback_markup("555", "Hello.") is None


def test_feedback_markup_absent_when_disabled(store):
    recorder = ConversationRecorder(store, feedback_buttons=False)
    recorder.record_chat("hi", "Hello.")
    recorder.record_turn("hi", "Hello.", route="chat", user_id="555")
    assert recorder.feedback_markup("555", "Hello.") is None


def test_handle_callback_latest_press_wins(store, recorder):
    # Changing your mind (👍 then 👎) keeps ONE row holding the latest choice.
    eid = recorder.record_chat("hi", "Hello.")
    recorder.handle_callback("555", f"fb:{eid}:1")
    recorder.handle_callback("555", f"fb:{eid}:0")
    fb = store.feedback_for(eid)
    assert [(f["kind"], f["signal"], f["source"]) for f in fb] == [
        ("explicit", -1, "telegram_button"),
    ]


def test_handle_callback_double_press_records_once(store, recorder):
    eid = recorder.record_chat("hi", "Hello.")
    recorder.handle_callback("555", f"fb:{eid}:0")
    recorder.handle_callback("555", f"fb:{eid}:0")
    fb = store.feedback_for(eid)
    assert len(fb) == 1
    assert fb[0]["signal"] == -1


def test_handle_callback_ignores_malformed_data(store, recorder):
    for junk in ("", "fb:", "fb:notanint:1", "other:1:1", None):
        recorder.handle_callback("555", junk)
    assert store.counts()["feedback"] == 0


# ------------------------------------------------------------- provider hook

class _FakeFridayStore:
    def __init__(self):
        self.turns = []

    def log_turn(self, role, content):
        self.turns.append((role, content))
        return len(self.turns)


class _FakeContextBuilder:
    def query_fingerprint(self, text):
        return f"fp:{text}"

    def build_context(self, text):
        return {"retrieved_fact_keys": []}

    def format_system_prompt(self, system_prompt, context):
        return system_prompt + "\n<context/>"


class _FakeExtractor:
    def is_correction(self, text):
        return False

    def extract(self, user_input, response):
        return []


class _FakeProvider(LLMProvider):
    def generate_response(self, user_input: str) -> str:  # pragma: no cover
        return "unused"

    def get_name(self) -> str:
        return "Fake (test)"


def _make_provider(recorder=None, ephemeral=False):
    provider = _FakeProvider(LLMConfig(ephemeral=True), PersonalityConfig())
    if not ephemeral:
        provider.store = _FakeFridayStore()
        provider.context_builder = _FakeContextBuilder()
        provider.extractor = _FakeExtractor()
    provider.research_recorder = recorder
    return provider


def test_prepare_request_returns_augmented_prompt_non_ephemeral(store, recorder):
    """Regression: a bad edit once dedented the ephemeral-branch return in
    _prepare_request so it ran unconditionally, crashing every production
    (non-ephemeral) reply with UnboundLocalError and dead-coding the context
    build. Drive the real method on both paths."""
    provider = _make_provider(recorder)  # non-ephemeral: fake context builder attached
    cached, augmented = provider._prepare_request("hello there")
    assert cached is None
    assert augmented.endswith("<context/>")  # context build path actually ran
    assert provider._last_augmented_prompt == augmented

    ephemeral = _make_provider(recorder, ephemeral=True)
    cached, augmented = ephemeral._prepare_request("hello there")
    assert cached is None
    assert augmented == ephemeral.system_prompt  # ephemeral branch: plain persona
    assert ephemeral._last_augmented_prompt == augmented


def test_generate_response_end_to_end_non_ephemeral(store, recorder):
    """The full provider path: _prepare_request -> API call -> _record_exchange
    records once with the augmented snapshot."""
    provider = _make_provider(recorder)
    cached, augmented = provider._prepare_request("what's for dinner")
    provider.conversation_history = [
        {"role": "user", "content": "what's for dinner"},
        {"role": "assistant", "content": "Pasta, sir."},
    ]
    provider._record_exchange("what's for dinner", "Pasta, sir.")
    row = store.get_exchange(1)
    assert row["context_snapshot"]["system_prompt"] == augmented
    assert row["context_snapshot"]["system_prompt"].endswith("<context/>")


def test_provider_records_exchange_with_snapshot(store, recorder):
    provider = _make_provider(recorder)
    provider._last_augmented_prompt = "AUGMENTED PROMPT"
    provider.conversation_history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Good evening, sir."},
    ]
    provider._record_exchange("hello", "Good evening, sir.")

    assert store.counts()["exchanges"] == 1
    row = store.get_exchange(1)
    assert row["route"] == "chat"
    assert row["model"] == "Fake (test)"
    assert row["memory_turn_id"] == 1  # id of the user turn in the fake store
    snap = row["context_snapshot"]
    assert snap["system_prompt"] == "AUGMENTED PROMPT"
    # Snapshot ends with the user message; the reply is excluded.
    assert snap["messages"] == [{"role": "user", "content": "hello"}]


def test_provider_recorder_failure_does_not_raise(store):
    class ExplodingRecorder:
        def record_chat(self, *a, **k):
            raise RuntimeError("boom")

    provider = _make_provider(ExplodingRecorder())
    provider.conversation_history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello."},
    ]
    provider._record_exchange("hi", "Hello.")  # must not raise
    assert provider.store.turns  # normal persistence still happened


def test_ephemeral_provider_records_nothing(store, recorder):
    provider = _make_provider(recorder, ephemeral=True)
    provider._record_exchange("hi", "Hello.")
    assert store.counts()["exchanges"] == 0
