"""Records layer: eval series, run history, artifact growth, transcript search."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pytest

from introspection import records
from research.db import ResearchStore

NOW = time.time()
NOW_DT = datetime.fromtimestamp(NOW)

CSV_HEADER = ("date,run_id,arm,opponent,artifact_version,judge,split,"
              "n_prompts,n_decisive,wins,losses,win_rate,p_value,"
              "n_control,control_win_rate,arm_style,opponent_style\n")


@pytest.fixture
def eval_csv(tmp_path):
    path = tmp_path / "eval.csv"
    path.write_text(
        CSV_HEADER
        + "20260810,1,memory,base,v20260810,sonnet:x,curated,32,20,9,11,0.450,0.800,6,0.500,0.95,0.96\n"
        + "20260817,2,memory,base,v20260817,sonnet:x,curated,32,22,11,11,0.500,0.700,6,0.500,0.95,0.95\n"
        + "20260824,3,memory,base,v20260824,sonnet:x,curated,32,24,14,10,0.583,0.041,6,0.500,0.96,0.95\n"
        + "20260824,3,memory,base,v20260824,fake,curated,4,2,2,0,1.000,0.100,0,,1.0,1.0\n")
    return path


def test_eval_history_series_sorted_and_fake_excluded(eval_csv):
    hist = records.eval_history(eval_csv)
    assert hist["available"] is True
    series = hist["series"]["memory/curated"]
    assert [e["date"] for e in series] == ["20260810", "20260817", "20260824"]
    assert series[-1]["win_rate"] == pytest.approx(0.583)
    assert all(e["judge"] != "fake" for e in series)
    assert hist["eval_dates"] == ["20260810", "20260817", "20260824"]


def test_eval_history_bar_verdict(eval_csv):
    bar = records.eval_history(eval_csv)["bars"]["memory"]
    # Latest row passes 1-5 but the prior week's didn't → condition 6 fails.
    assert bar["improved"] is False
    numbers = {c["number"]: c["passed"] for c in bar["conditions"]}
    assert numbers[1] is True and numbers[6] is False
    assert "fails" in bar["summary"]


def test_eval_history_missing_csv_degrades(tmp_path):
    assert records.eval_history(tmp_path / "none.csv") == {"available": False}


def test_nightly_reports_reads_newest(tmp_path):
    nightly = tmp_path / "nightly"
    nightly.mkdir()
    for day in ("20260820", "20260821", "20260822"):
        (nightly / f"{day}.md").write_text(f"# Nightly {day}\n")
    out = records.nightly_reports(tmp_path, limit=2)
    assert [r["name"] for r in out["reports"]] == ["20260821.md", "20260822.md"]
    assert "Nightly 20260822" in out["reports"][-1]["text"]


@pytest.fixture
def research_db(tmp_path):
    path = tmp_path / "research.db"
    store = ResearchStore(str(path))
    for days_ago, stages in ((3, {"harvest": "ok (1s)", "train": "ok (200s)"}),
                             (2, {"harvest": "ok (1s)",
                                  "train": "FAILED: RuntimeError: oom"}),
                             (1, {"harvest": "ok (1s)", "train": "ok (210s)"})):
        started = NOW - days_ago * 86400
        store.execute(
            "INSERT INTO runs (started_ts, finished_ts, stage_status)"
            " VALUES (?, ?, ?)", (started, started + 300, json.dumps(stages)))
    yday_start, _ = records.past_date("yesterday", NOW_DT)
    for ts, text, reply in ((NOW, "shall we plan the dentist visit",
                             "Certainly, sir."),
                            (yday_start + 10 * 3600, "turn on the lights",
                             "Done, sir."),
                            (yday_start + 11 * 3600,
                             "what about the dentist appointment",
                             "It is on Thursday, sir.")):
        store.execute(
            "INSERT INTO exchanges (ts, user_text, reply_text, route, channel)"
            " VALUES (?, ?, ?, 'chat', 'text')", (ts, text, reply))
    ex_id = store.query("SELECT id FROM exchanges LIMIT 1")[0]["id"]
    store.add_feedback(ex_id, kind="explicit", signal=1, source="telegram:button")
    store.close()
    return path


def test_runs_history_aggregates(research_db):
    hist = records.runs_history(research_db)
    assert hist["total"] == 3
    assert hist["runs_with_failures"] == 1
    assert hist["most_common_failing_stage"] == "train"
    assert hist["runs"][0]["run_id"] == 3          # newest first
    assert hist["runs"][1]["failed_stages"] == ["train"]
    assert hist["runs"][0]["duration_s"] == 300


def test_artifact_history_growth_series(tmp_path):
    from research.provenance import write_manifest

    art = tmp_path / "artifacts"
    for version, n_train in (("v20260810", 20), ("v20260824", 45)):
        vdir = art / "lora" / version
        vdir.mkdir(parents=True)
        write_manifest(vdir, "lora", dataset={"n_train": n_train},
                       params={"iters": 4 * n_train})
    (art / "lora" / "v20260824" / "GATED").write_text("style 0.4\n")
    (art / "lora" / "current").write_text("v20260810\n")

    hist = records.artifact_history("lora", art)
    assert [v["version"] for v in hist["versions"]] == ["v20260810", "v20260824"]
    assert [v["dataset"]["n_train"] for v in hist["versions"]] == [20, 45]
    assert hist["versions"][0]["is_current"] is True
    assert hist["versions"][1]["gated"] is True
    assert records.artifact_history("lora", tmp_path / "none")["available"] is False


def test_feedback_stats_counts(research_db):
    stats = records.feedback_stats(research_db, days=30, now=NOW)
    assert stats["positive"] == 1 and stats["negative"] == 0
    assert stats["by_source"] == [{"source": "telegram:button", "signal": 1,
                                   "count": 1}]
    assert len(stats["by_day"]) == 1


def test_conversation_search_keyword_and_window(research_db):
    out = records.conversation_search(research_db, query="dentist")
    assert out["total_matches"] == 2
    assert [m["user"] for m in out["matches"]] == [
        "what about the dentist appointment", "shall we plan the dentist visit"]

    yesterday = records.past_date("yesterday", NOW_DT)
    out = records.conversation_search(research_db, since_ts=yesterday[0],
                                      until_ts=yesterday[1])
    assert out["total_matches"] == 2          # the two ~1-day-old exchanges
    assert all("dentist" in m["user"] or "lights" in m["user"]
               for m in out["matches"])


def test_conversation_search_truncates_excerpts(tmp_path):
    path = tmp_path / "research.db"
    store = ResearchStore(str(path))
    store.execute(
        "INSERT INTO exchanges (ts, user_text, reply_text, route) VALUES (?, ?, ?, 'chat')",
        (NOW, "x" * 1000, "y" * 1000))
    store.close()
    match = records.conversation_search(path)["matches"][0]
    assert len(match["user"]) == records.EXCERPT_CHARS
    assert len(match["reply"]) == records.EXCERPT_CHARS


def test_conversation_search_falls_back_to_memory_summaries(tmp_path):
    import sqlite3

    mem = tmp_path / "memory.db"
    conn = sqlite3.connect(mem)
    conn.execute("CREATE TABLE conversation_summaries (id INTEGER PRIMARY KEY,"
                 " summary TEXT, turn_range_start INT, turn_range_end INT,"
                 " created_at TEXT DEFAULT (datetime('now')))")
    conn.execute("INSERT INTO conversation_summaries (summary) VALUES"
                 " ('Discussed the dentist and the coffee machine')")
    conn.commit()
    conn.close()
    out = records.conversation_search(tmp_path / "none.db", query="dentist",
                                      memory_db=mem)
    assert out["source"] == "summaries"
    assert out["total_matches"] == 1


def test_everything_degrades_without_creating_files(tmp_path):
    root = tmp_path / "nothing"
    assert records.runs_history(root / "r.db")["available"] is False
    assert records.feedback_stats(root / "r.db")["available"] is False
    assert records.conversation_search(root / "r.db")["available"] is False
    assert records.nightly_reports(root)["available"] is False
    assert not root.exists()


def test_past_date_resolves_backward():
    now = datetime(2026, 8, 27, 15, 0)          # a Thursday
    start, end = records.past_date("yesterday", now)
    assert datetime.fromtimestamp(start).date().isoformat() == "2026-08-26"
    assert end - start == 86400

    start, _ = records.past_date("last tuesday", now)
    assert datetime.fromtimestamp(start).date().isoformat() == "2026-08-25"

    # Same weekday as today → the previous one, never today.
    start, _ = records.past_date("on thursday", now)
    assert datetime.fromtimestamp(start).date().isoformat() == "2026-08-20"

    start, end = records.past_date("last week", now)
    assert datetime.fromtimestamp(start).date().isoformat() == "2026-08-17"
    assert (end - start) == 7 * 86400

    start, _ = records.past_date("2026-08-01", now)
    assert datetime.fromtimestamp(start).date().isoformat() == "2026-08-01"

    assert records.past_date("the mitochondria", now) is None
    assert records.past_date("", now) is None
