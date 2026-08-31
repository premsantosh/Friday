"""The REAL ContextBuilder against a real tmp FridayStore.

The agent tests substitute FakeContextBuilder, which is exactly where the
render-relevant-facts and search bugs hid; this file carries the correctness
burden for the real builder.
"""

from __future__ import annotations

import pytest

from memory.cache import FridayCache
from memory.context_builder import ContextBuilder, should_cache_response
from memory.store import FridayStore


@pytest.fixture
def store(tmp_path):
    return FridayStore(db_path=str(tmp_path / "memory.db"))


@pytest.fixture
def builder(store):
    return ContextBuilder(FridayCache(), store)


def test_relevant_facts_are_rendered_into_the_prompt(builder, store):
    store.remember(key="espresso_preference", value="double shot, oat milk",
                   category="preference", confidence=0.9)
    # "before" gates the HISTORY branch; "espresso" is the search term.
    ctx = builder.build_context("what did i say about the espresso order before")
    assert ctx["relevant_facts"], "search_facts found nothing"
    prompt = builder.format_system_prompt("BASE", ctx)
    assert "<relevant_facts>" in prompt
    assert "espresso_preference: double shot, oat milk" in prompt


def test_low_confidence_facts_stay_out_of_personal_context(builder, store):
    store.remember(key="wife_name", value="Beatrice", category="personal",
                   confidence=0.9)
    store.remember(key="country_of_origin", value="unknown", category="personal",
                   confidence=0.3)
    ctx = builder.build_context("what do you know about my family")
    assert ctx["personal"] == {"wife_name": "Beatrice"}
    prompt = builder.format_system_prompt("BASE", ctx)
    assert "country_of_origin" not in prompt


def test_should_cache_response_accepts_self_contained_questions():
    assert should_cache_response("what tea do I like?")
    assert should_cache_response("how was last night's lora run")
    assert should_cache_response("what's the capital of france called")


def test_should_cache_response_rejects_context_dependent_followups():
    # The exact production failure: same short follow-up twice within the TTL
    # got the identical earlier reply in a different conversation.
    assert not should_cache_response("give me more details")
    assert not should_cache_response("yes")
    assert not should_cache_response("why?")
    assert not should_cache_response("tell me about that")
    assert not should_cache_response("do it again")
