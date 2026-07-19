"""Tests for implicit-signal miners (research/miners.py). Pure functions."""

from __future__ import annotations

from research.miners import mine_all, mine_corrections, mine_rephrases


def _ex(id, ts, text, user="u1"):
    return {"id": id, "ts": ts, "user_id": user, "user_text": text}


# ------------------------------------------------------------------ rephrases

def test_rephrase_flags_first_of_similar_pair():
    exchanges = [
        _ex(1, 100.0, "book a table at the italian place tonight"),
        _ex(2, 130.0, "book a table at the italian restaurant for tonight"),
    ]
    sigs = mine_rephrases(exchanges)
    assert len(sigs) == 1
    assert sigs[0].exchange_id == 1
    assert sigs[0].signal == -1
    assert sigs[0].source == "miner:rephrase"


def test_rephrase_ignores_identical_repeat():
    exchanges = [_ex(1, 100.0, "turn on the lights"), _ex(2, 110.0, "turn on the lights")]
    assert mine_rephrases(exchanges) == []


def test_rephrase_ignores_dissimilar_followup():
    exchanges = [_ex(1, 100.0, "what time is it"), _ex(2, 110.0, "make me a coffee")]
    assert mine_rephrases(exchanges) == []


def test_rephrase_ignores_outside_window():
    exchanges = [
        _ex(1, 100.0, "book a table at the italian place tonight"),
        _ex(2, 100.0 + 600, "book a table at the italian restaurant tonight"),
    ]
    assert mine_rephrases(exchanges) == []


def test_rephrase_ignores_different_users():
    exchanges = [
        _ex(1, 100.0, "book a table at the italian place", user="u1"),
        _ex(2, 110.0, "book a table at the italian place now", user="u2"),
    ]
    assert mine_rephrases(exchanges) == []


# ---------------------------------------------------------------- corrections

def test_correction_opener_flags_previous_exchange():
    exchanges = [
        _ex(1, 100.0, "remind me about mum's birthday"),
        _ex(2, 120.0, "no, I said next Friday"),
    ]
    sigs = mine_corrections(exchanges)
    assert len(sigs) == 1
    assert sigs[0].exchange_id == 1
    assert sigs[0].signal == -1


def test_correction_mid_sentence_not_flagged():
    exchanges = [
        _ex(1, 100.0, "what's on my calendar"),
        _ex(2, 120.0, "I know there's no meeting, just checking"),
    ]
    assert mine_corrections(exchanges) == []


# ------------------------------------------------------------------- mine_all

def test_mine_all_dedupes_per_exchange_and_source():
    # A rephrase that also opens with a correction marker: two different
    # sources may both flag exchange 1, but each source only once.
    exchanges = [
        _ex(1, 100.0, "set an alarm for six tomorrow"),
        _ex(2, 110.0, "no, set an alarm for six thirty tomorrow"),
    ]
    sigs = mine_all(exchanges)
    keys = [(s.exchange_id, s.source) for s in sigs]
    assert len(keys) == len(set(keys))
    assert {s.source for s in sigs} == {"miner:rephrase", "miner:correction"}
