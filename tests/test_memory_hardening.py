"""FridayStore/extractor/_extract_and_store hardening, on real tmp sqlite.

Regression suite for the 2026-08-30 production incident: the fact-extraction
daemon thread died on a list-valued fact (sqlite3.ProgrammingError), and junk
facts (current_time, salutation) were stored and replayed as stale truth.
"""

from __future__ import annotations

import json

import pytest

from config import LLMConfig, PersonalityConfig
from llm.providers import JUNK_FACT_KEYS, LLMProvider
from memory.extractor import FactExtractor
from memory.store import FridayStore


@pytest.fixture
def store(tmp_path):
    return FridayStore(db_path=str(tmp_path / "memory.db"))


# ------------------------------------------------------------------ store

def test_remember_coerces_list_value(store):
    store.remember(key="travelers", value=["Alistair", "Beatrice"], category="personal",
                   confidence=0.9)
    assert store.recall("travelers") == "Alistair, Beatrice"


def test_remember_coerces_dict_and_scalar_values(store):
    store.remember(key="trip", value={"where": "Japan"}, category="personal")
    assert json.loads(store.recall("trip")) == {"where": "Japan"}
    store.remember(key="party_size", value=2, category="routine")
    assert store.recall("party_size") == "2"


def test_search_facts_word_splits_the_query(store):
    store.remember(key="espresso_preference", value="double shot, oat milk",
                   category="preference", confidence=0.9)
    # The old whole-sentence LIKE never matched anything.
    hits = store.search_facts("what did I say about the espresso machine before")
    assert [k for k, _, _ in hits] == ["espresso_preference"]


def test_search_facts_confidence_floor(store):
    store.remember(key="espresso_guess", value="latte?", category="preference",
                   confidence=0.4)
    assert store.search_facts("espresso order") == []
    store.bump_confidence("espresso_guess", delta=0.2)
    assert [k for k, _, _ in store.search_facts("espresso order")] == ["espresso_guess"]


def test_search_facts_caps_results(store):
    for i in range(10):
        store.remember(key=f"espresso_fact_{i}", value="x", category="general",
                       confidence=0.9)
    assert len(store.search_facts("espresso")) == 6


def test_recall_by_category_confidence_floor(store):
    store.remember(key="wife_name", value="Beatrice", category="personal", confidence=0.9)
    store.remember(key="country_of_origin", value="unknown", category="personal",
                   confidence=0.3)
    assert store.recall_by_category("personal") == {"wife_name": "Beatrice"}
    assert "country_of_origin" in store.recall_by_category("personal", min_confidence=0.0)


# -------------------------------------------------------------- extractor

def _extractor_with_response(monkeypatch, payload) -> FactExtractor:
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": json.dumps(payload)}

    import memory.extractor as mod
    monkeypatch.setattr(mod.requests, "post", lambda *a, **kw: _Resp())
    return FactExtractor()


def test_extractor_coerces_list_values_and_drops_bad_shapes(monkeypatch):
    extractor = _extractor_with_response(monkeypatch, [
        {"key": "travelers", "value": ["Alistair", "Beatrice"], "category": "personal"},
        {"key": "trip", "value": {"where": "Japan"}, "category": "personal"},
        {"key": 42, "value": "x", "category": "personal"},
        {"key": "wife_name", "value": "Beatrice", "category": "personal"},
    ])
    facts = extractor.extract("my wife is Beatrice", "Noted, sir.")
    assert [(f["key"], f["value"]) for f in facts] == [
        ("travelers", "Alistair, Beatrice"),
        ("wife_name", "Beatrice"),
    ]


# ------------------------------------------------------- _extract_and_store

class _Provider(LLMProvider):
    def generate_response(self, user_input, **kwargs):  # pragma: no cover
        return "unused"

    def get_name(self):
        return "Test"


def _provider(store, extractor) -> LLMProvider:
    p = _Provider(LLMConfig(ephemeral=True), PersonalityConfig())
    p.store = store
    p.extractor = extractor
    return p


class _StubExtractor:
    def __init__(self, facts):
        self._facts = facts

    def extract(self, user_input, response):
        return self._facts


def test_extract_and_store_skips_junk_and_low_confidence(store):
    provider = _provider(store, _StubExtractor([
        {"key": "current_time", "value": "half past eleven", "category": "routine",
         "confidence": 0.9},
        {"key": "Salutation", "value": "Yo", "category": "general", "confidence": 0.9},
        {"key": "hunch", "value": "?", "category": "general", "confidence": 0.2},
        {"key": "wife_name", "value": "Beatrice", "category": "personal",
         "confidence": 0.9},
    ]))
    provider._extract_and_store("hi", "hello")
    assert store.recall("wife_name") == "Beatrice"
    assert store.recall("current_time") is None
    assert store.recall("salutation") is None
    assert store.recall("hunch") is None
    assert provider.stats["facts_stored"] == 1


def test_extract_and_store_survives_one_bad_fact(store):
    class _ExplodingStore:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0

        def remember(self, **kw):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            self.inner.remember(**kw)

    exploding = _ExplodingStore(store)
    provider = _provider(exploding, _StubExtractor([
        {"key": "first", "value": "x", "category": "general", "confidence": 0.9},
        {"key": "second", "value": "y", "category": "general", "confidence": 0.9},
    ]))
    provider._extract_and_store("hi", "hello")   # must not raise
    assert store.recall("second") == "y"         # the batch survived the bad fact


def test_extract_and_store_never_raises_on_extractor_failure(store):
    class _Broken:
        def extract(self, user_input, response):
            raise RuntimeError("ollama down")

    provider = _provider(store, _Broken())
    provider._extract_and_store("hi", "hello")   # must not raise


def test_junk_keys_cover_the_observed_offenders():
    for key in ("current_time", "salutation", "inquiry_type"):
        assert key in JUNK_FACT_KEYS
