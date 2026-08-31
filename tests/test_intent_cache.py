"""IntentCache against a real ChromaDB collection in tmp_path."""

from __future__ import annotations

import pytest

pytest.importorskip("chromadb")

from core.intent_cache import IntentCache


def test_delete_workflow_removes_only_that_workflows_entries(tmp_path):
    cache = IntentCache(path=str(tmp_path / "intents"), similarity_threshold=0.75)
    cache.store("How was last night's LoRA run?", "self_status", {})
    cache.store("yeah I'm talking about the LoRA training cycle", "self_status", {})
    cache.store("turn off the kitchen lights", "hue_lights", {"action": "off"})
    assert cache.count() == 3
    assert cache.query("How was last night's LoRA run?")[0] == "self_status"

    assert cache.delete_workflow("self_status") == 2
    assert cache.count() == 1
    assert cache.query("How was last night's LoRA run?") is None
    assert cache.query("turn off the kitchen lights") == ("hue_lights", {"action": "off"})
    assert cache.delete_workflow("self_status") == 0    # idempotent


def test_expired_entry_is_lazily_deleted_on_query(tmp_path):
    cache = IntentCache(path=str(tmp_path / "intents"), ttl_days=30)
    cache.store("turn off the kitchen lights", "hue_lights", {"action": "off"})

    # Backdate created_at past the TTL (metadata is the source of truth).
    from datetime import datetime, timedelta
    collection = cache._get_collection()
    row = collection.get(include=["metadatas"])
    old = (datetime.now() - timedelta(days=31)).isoformat()
    collection.update(ids=row["ids"],
                      metadatas=[{**row["metadatas"][0], "created_at": old}])

    assert cache.query("turn off the kitchen lights") is None
    assert cache.count() == 0                           # lazy delete happened


def test_fresh_entry_survives_query_with_ttl(tmp_path):
    cache = IntentCache(path=str(tmp_path / "intents"), ttl_days=30)
    cache.store("turn off the kitchen lights", "hue_lights", {"action": "off"})
    assert cache.query("turn off the kitchen lights") == ("hue_lights", {"action": "off"})
    assert cache.count() == 1
