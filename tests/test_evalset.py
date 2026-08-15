"""Eval set loading: placeholder refusal and honest harvested categories."""

from __future__ import annotations

import pytest
import yaml

from research.db import ResearchStore
from research.evalset import (CURATED_PATH, count_placeholders, load_curated,
                              load_harvested)


def _write_yaml(path, prompts):
    path.write_text(yaml.safe_dump({"prompts": prompts}))
    return path


def test_load_curated_refuses_placeholder_annotations(tmp_path):
    """Annotations reach the judge verbatim as facts about the user.

    A FILL-IN placeholder tells the judge that a known fact is the string
    "FILL-IN: your actual coffee order", so verdicts on that probe are noise
    presented as measurement.
    """
    path = _write_yaml(tmp_path / "c.yaml", [
        {"id": "pref-01", "category": "preference_recall",
         "prompt": "coffee order?", "annotations": "FILL-IN: your coffee order."},
        {"id": "ctrl-01", "category": "generic_control",
         "prompt": "capital of France?", "annotations": "Paris."},
    ])

    with pytest.raises(ValueError) as e:
        load_curated(path)

    assert "pref-01" in str(e.value)
    assert "ctrl-01" not in str(e.value), "only the offending ids are named"


def test_load_curated_allows_placeholders_when_asked(tmp_path):
    path = _write_yaml(tmp_path / "c.yaml", [
        {"id": "pref-01", "category": "preference_recall",
         "prompt": "coffee order?", "annotations": "FILL-IN: your coffee order."},
    ])
    assert len(load_curated(path, allow_placeholders=True)) == 1


def test_load_curated_accepts_filled_annotations(tmp_path):
    path = _write_yaml(tmp_path / "c.yaml", [
        {"id": "pref-01", "category": "preference_recall",
         "prompt": "coffee order?", "annotations": "Flat white, oat milk."},
    ])
    prompts = load_curated(path)
    assert prompts[0].annotations == "Flat white, oat milk."


def test_count_placeholders_tracks_the_outstanding_chore(tmp_path):
    path = _write_yaml(tmp_path / "c.yaml", [
        {"id": "a", "category": "style", "prompt": "x", "annotations": "FILL-IN: y"},
        {"id": "b", "category": "style", "prompt": "x", "annotations": "done"},
    ])
    assert count_placeholders(path) == 1
    assert count_placeholders(tmp_path / "missing.yaml") == 0


def test_shipped_curated_file_is_still_valid_yaml():
    """The real file must always parse, placeholders or not."""
    prompts = load_curated(CURATED_PATH, allow_placeholders=True)
    assert len(prompts) >= 15
    assert count_placeholders(CURATED_PATH) == len(
        [p for p in prompts if "FILL-IN" in p.annotations])


def test_harvested_prompts_are_labelled_harvested(tmp_path):
    """Not 'preference_recall': these are whatever the user happened to say."""
    store = ResearchStore(str(tmp_path / "r.db"))
    try:
        store.record_exchange("what's for dinner", "Chicken, sir.", route="chat",
                              ts=100.0)
        store.record_exchange("lights on", "Done, sir.", route="keyword:hue",
                              ts=200.0)

        prompts = load_harvested(store, after_ts=0.0)

        assert len(prompts) == 1, "workflow routes are not free conversation"
        assert prompts[0].category == "harvested"
        assert prompts[0].source == "harvested"
        assert prompts[0].annotations == ""
    finally:
        store.close()


def test_harvested_respects_the_temporal_split(tmp_path):
    store = ResearchStore(str(tmp_path / "r.db"))
    try:
        store.record_exchange("old", "reply", route="chat", ts=100.0)
        store.record_exchange("new", "reply", route="chat", ts=300.0)

        prompts = load_harvested(store, after_ts=200.0)

        assert [p.prompt for p in prompts] == ["new"]
    finally:
        store.close()
