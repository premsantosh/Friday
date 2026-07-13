"""Tests for the memory pipeline: store, context builder, privacy gating,
history hydration/trim, and the correction heuristic."""

import threading

import pytest

from memory.cache import FridayCache
from memory.context_builder import ContextBuilder
from memory.extractor import FactExtractor
from memory.store import FridayStore


@pytest.fixture
def store(tmp_path):
    return FridayStore(db_path=str(tmp_path / "memory.db"))


@pytest.fixture
def builder(store):
    return ContextBuilder(FridayCache(), store)


# ------------------------------------------------------------------- store

def test_search_facts_matches_keywords_inside_sentences(store):
    store.remember("coffee_order", "oat milk flat white", category="preference")
    results = store.search_facts("what did i say my coffee order was")
    assert any(k == "coffee_order" for k, _, _ in results)


def test_search_facts_respects_confidence_floor(store):
    store.remember("guessed_fact", "probably likes jazz", confidence=0.4)
    assert store.search_facts("jazz", min_confidence=0.6) == []
    assert store.search_facts("jazz") != []


def test_drop_confidence_only_deletes_that_key(store):
    store.remember("a", "1", confidence=0.15)
    store.remember("b", "2", confidence=0.15)
    store.drop_confidence("a")  # 0.15 - 0.3 → below floor, deleted
    assert store.recall("a") is None
    assert store.recall("b") == "2"


def test_store_is_thread_safe_under_concurrent_writes(store):
    errors = []

    def writer(i):
        try:
            for j in range(50):
                store.remember(f"k{i}_{j}", "v")
                store.log_turn("user", f"msg {i}-{j}")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert store.count_turns() == 200


def test_summary_prune_deletes_exact_rows(store):
    for i in range(10):
        store.log_turn("user", f"u{i}")
    turns = store.get_oldest_turns(4)
    ids = [t["id"] for t in turns]
    store.save_summary("summary of first four", ids[0], ids[-1])
    store.delete_turns(ids)
    assert store.count_turns() == 6
    assert store.get_summaries() == ["summary of first four"]


# ------------------------------------------------------------- context builder

def test_private_context_withheld_for_cloud_prompts(store, builder):
    store.remember("name", "Prem", category="personal")
    ctx = builder.build_context("what's my name", include_private=False)
    assert "personal" not in ctx
    assert "preferences" not in ctx
    assert ctx["retrieved_fact_keys"] == []


def test_private_context_included_locally(store, builder):
    store.remember("name", "Prem", category="personal")
    ctx = builder.build_context("what's my name", include_private=True)
    assert ctx["personal"] == {"name": "Prem"}


def test_relevant_facts_are_rendered_into_prompt(store, builder):
    store.remember("coffee_order", "oat milk flat white", category="preference")
    ctx = builder.build_context("remember my coffee order?")
    prompt = builder.format_system_prompt("BASE", ctx)
    assert "oat milk flat white" in prompt


def test_volatile_queries_are_not_cacheable(builder):
    assert not builder.is_cacheable("what time is it")
    assert not builder.is_cacheable("is the front door locked")
    assert builder.is_cacheable("tell me about the roman empire")


def test_low_confidence_facts_not_injected(store, builder):
    store.remember("guess", "maybe vegetarian", category="preference", confidence=0.4)
    ctx = builder.build_context("what do i usually eat")
    assert ctx.get("preferences") == {}


# ------------------------------------------------------------------ extractor

def test_correction_requires_leading_or_explicit_phrase():
    ex = FactExtractor()
    assert ex.is_correction("No, I take oat milk")
    assert ex.is_correction("actually it's Tuesday")
    assert ex.is_correction("that's wrong")
    # A "no" or "not" buried mid-sentence is not a correction signal.
    assert not ex.is_correction("I know, right")
    assert not ex.is_correction("is there no coffee left")
    assert not ex.is_correction("play something not too loud")


# ------------------------------------------------------- history trim / hydrate

def _bare_provider():
    """A provider instance without running __init__ (no network, no config)."""
    from llm.providers import LLMProvider

    class _Stub(LLMProvider):
        def generate_response(self, user_input):  # pragma: no cover
            raise NotImplementedError

        def get_name(self):  # pragma: no cover
            return "stub"

    return _Stub.__new__(_Stub)


def test_trim_history_never_starts_with_assistant():
    p = _bare_provider()

    class Cfg:
        max_history = 10

    p.config = Cfg()
    # 11 alternating messages starting with user → naive slice would start
    # with an assistant turn and the Anthropic API would reject the request.
    p.conversation_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
        for i in range(11)
    ]
    p._trim_history()
    assert p.conversation_history[0]["role"] == "user"
    assert len(p.conversation_history) <= 10


def test_hydrate_history_restores_alternating_turns(store):
    store.log_turn("assistant", "orphaned greeting")  # must be skipped
    store.log_turn("user", "hello")
    store.log_turn("assistant", "good day, sir")
    store.log_turn("user", "dangling user turn")  # dropped: history must end on assistant

    p = _bare_provider()

    class Cfg:
        max_history = 10

    p.config = Cfg()
    p.store = store
    p._hydrate_history()

    assert p.conversation_history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "good day, sir"},
    ]
