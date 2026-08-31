"""`research rate --mode arm-base`: the pre-registered human anchor."""

from __future__ import annotations

import argparse
import random
import time

import pytest

from research.cli import _rate_arm_base
from research.db import ResearchStore


@pytest.fixture
def store(tmp_path):
    s = ResearchStore(str(tmp_path / "research.db"))
    yield s
    s.close()


def _seed_run(store) -> int:
    """A judged run: one replay pair (exchange 1) and one curated pair."""
    run_id = store.execute(
        "INSERT INTO runs (started_ts, stage_status) VALUES (?, '{}')",
        (time.time(),))
    eid = store.record_exchange("what tea do I like?", "Earl Grey, sir.",
                                route="chat")
    store.add_shadow_response(eid, arm="memory", mode="replay",
                              response_text="Earl Grey, as always, sir.")
    store.add_shadow_response(eid, arm="base", mode="replay",
                              response_text="I do not know your tea, sir.")
    store.add_curated_response(run_id, "style-01", "memory", "Half past nine, sir.")
    store.add_curated_response(run_id, "style-01", "base", "It is 9:30 PM!")
    for pid in (str(eid), "style-01"):
        store.execute(
            "INSERT INTO eval_results (run_id, arm, prompt_id, opponent, winner,"
            " judge, scores, ts) VALUES (?, 'memory', ?, 'base', 'arm',"
            " 'sonnet:claude-sonnet-5', NULL, ?)",
            (run_id, pid, time.time()))
    return run_id


def _args(db, pairs=20):
    return argparse.Namespace(db=db, pairs=pairs, mode="arm-base")


def _drive(monkeypatch, answers):
    answers = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))


def test_ratings_deanonymize_correctly_for_both_side_orders(store, tmp_path, monkeypatch):
    run_id = _seed_run(store)
    # The presentation order is seeded per (run, arm, prompt): recompute it the
    # way the CLI does, then always answer "1" and check the mapped winner.
    pair_ids = [str(store.query("SELECT id FROM exchanges")[0]["id"]), "style-01"]
    expected = {}
    for pid in pair_ids:
        rng = random.Random(f"{run_id}:memory:{pid}")
        expected[pid] = "arm" if rng.random() < 0.5 else "base"

    _drive(monkeypatch, ["1", "1"])
    assert _rate_arm_base(_args(str(tmp_path / "research.db"))) == 0

    human = {r["prompt_id"]: r["winner"] for r in store.query(
        "SELECT prompt_id, winner FROM eval_results WHERE judge='human'")}
    assert human == expected


def test_rated_pairs_are_excluded_on_rerun(store, tmp_path, monkeypatch):
    _seed_run(store)
    _drive(monkeypatch, ["t", "t"])
    assert _rate_arm_base(_args(str(tmp_path / "research.db"))) == 0
    assert len(store.query("SELECT 1 FROM eval_results WHERE judge='human'")) == 2

    # Second session: nothing left; input must never be called.
    monkeypatch.setattr("builtins.input",
                        lambda prompt="": pytest.fail("asked to rate a rated pair"))
    assert _rate_arm_base(_args(str(tmp_path / "research.db"))) == 0
    assert len(store.query("SELECT 1 FROM eval_results WHERE judge='human'")) == 2


def test_skip_and_quit_write_nothing(store, tmp_path, monkeypatch):
    _seed_run(store)
    _drive(monkeypatch, ["s", "q"])
    assert _rate_arm_base(_args(str(tmp_path / "research.db"))) == 0
    assert store.query("SELECT 1 FROM eval_results WHERE judge='human'") == []


def test_no_judged_runs(store, tmp_path):
    assert _rate_arm_base(_args(str(tmp_path / "research.db"))) == 1
