"""Tests for artifact versioning, the prompt evolver, and the nightly loop.

Hermetic: tmp_path stores/artifacts, fake LLM functions, monkeypatched stages.
"""

from __future__ import annotations

import json

import pytest

from research import artifacts
from research.approaches import prompt_evolver
from research.db import ResearchStore
from research.nightly import run_nightly


@pytest.fixture
def store(tmp_path):
    s = ResearchStore(str(tmp_path / "research.db"))
    yield s
    s.close()


@pytest.fixture
def art_dir(tmp_path):
    return tmp_path / "artifacts"


# ------------------------------------------------------------------ artifacts

def test_version_lifecycle(art_dir):
    v1 = artifacts.new_version("prompt", "20260718", art_dir)
    assert v1.name == "v20260718"
    v2 = artifacts.new_version("prompt", "20260718", art_dir)
    assert v2.name == "v20260718-2"  # rerun the same day never overwrites

    assert artifacts.current_version("prompt", art_dir) is None
    artifacts.advance_current("prompt", v1.name, art_dir)
    assert artifacts.current_version("prompt", art_dir) == "v20260718"
    assert artifacts.current_path("prompt", art_dir) == v1

    # Revert is just re-pointing.
    artifacts.advance_current("prompt", v2.name, art_dir)
    artifacts.advance_current("prompt", v1.name, art_dir)
    assert artifacts.current_version("prompt", art_dir) == "v20260718"

    with pytest.raises(FileNotFoundError):
        artifacts.advance_current("prompt", "v-nonexistent", art_dir)

    assert artifacts.list_versions("prompt", art_dir) == ["v20260718", "v20260718-2"]


# ------------------------------------------------------------- prompt evolver

def _seed_chat(store, n=3):
    for i in range(n):
        eid = store.record_exchange(f"user msg {i}", f"reply {i}", route="chat",
                                    ts=1000.0 + i)
    store.add_feedback(eid, kind="explicit", signal=1, source="telegram_button")
    return eid


def test_evolve_writes_versioned_block_and_advances(store, art_dir):
    _seed_chat(store)

    def fake_llm(prompt):
        assert "user msg 0" in prompt and "+1" in prompt
        return json.dumps({"block": "- prefers oat milk", "changelog": "added milk"})

    version = prompt_evolver.evolve(store, since_ts=0.0, date_str="20260718",
                                    llm_fn=fake_llm, artifacts_dir=art_dir)
    assert version == "v20260718"
    assert prompt_evolver.load_current_block(art_dir) == "- prefers oat milk"
    assert prompt_evolver.system_block(art_dir) == "LEARNED PREFERENCES:\n- prefers oat milk"
    vdir = artifacts.arm_dir("prompt", art_dir) / version
    assert (vdir / "changelog.txt").read_text().strip() == "added milk"
    assert "+- prefers oat milk" in (vdir / "diff.patch").read_text()


def test_evolve_skips_without_data(store, art_dir):
    version = prompt_evolver.evolve(store, since_ts=0.0, date_str="20260718",
                                    llm_fn=lambda p: "unused", artifacts_dir=art_dir)
    assert version is None


def test_evolve_keeps_current_on_unparseable_output(store, art_dir):
    _seed_chat(store)
    prompt_evolver.evolve(store, since_ts=0.0, date_str="20260717",
                          llm_fn=lambda p: json.dumps({"block": "- v1", "changelog": "x"}),
                          artifacts_dir=art_dir)
    version = prompt_evolver.evolve(store, since_ts=0.0, date_str="20260718",
                                    llm_fn=lambda p: "not json at all",
                                    artifacts_dir=art_dir)
    assert version is None
    assert prompt_evolver.load_current_block(art_dir) == "- v1"


def test_evolve_enforces_size_cap(store, art_dir):
    _seed_chat(store)
    huge = "x" * (prompt_evolver.MAX_BLOCK_CHARS * 2)
    prompt_evolver.evolve(store, since_ts=0.0, date_str="20260718",
                          llm_fn=lambda p: json.dumps({"block": huge, "changelog": "big"}),
                          artifacts_dir=art_dir)
    assert len(prompt_evolver.load_current_block(art_dir)) == prompt_evolver.MAX_BLOCK_CHARS


def test_system_block_empty_when_no_artifact(art_dir):
    assert prompt_evolver.system_block(art_dir) == ""


# ---------------------------------------------------------------- nightly loop

def test_nightly_stage_failure_isolated(store, art_dir, monkeypatch):
    _seed_chat(store)

    def exploding(ctx):
        raise RuntimeError("boom")

    import research.nightly as nightly

    monkeypatch.setattr(
        nightly, "STAGES",
        [("harvest", nightly.stage_harvest),
         ("evolve", exploding),
         ("report", lambda ctx: "still ran")],
    )
    status = run_nightly(store, dry_run=True, date_str="20260718",
                         artifacts_dir=art_dir)
    assert status["harvest"].startswith("ok")
    assert status["evolve"].startswith("FAILED: RuntimeError")
    assert "still ran" in status["report"]

    # Status persisted to the runs table as JSON.
    row = store.conn.execute("SELECT stage_status FROM runs ORDER BY id DESC").fetchone()
    persisted = json.loads(row["stage_status"])
    assert persisted["evolve"].startswith("FAILED")


def test_nightly_harvest_mines_and_backs_up(store, art_dir):
    import time as _time
    now = _time.time()
    store.record_exchange("book the italian place", "Done.", route="chat",
                          user_id="u", ts=now - 60)
    store.record_exchange("no, I said the french place", "Apologies.", route="chat",
                          user_id="u", ts=now - 30)
    status = run_nightly(store, dry_run=True, date_str="20260718",
                         artifacts_dir=art_dir, stages=["harvest"])
    assert "1 new signals" in status["harvest"]
    assert (art_dir / "backups" / "research-20260718.db").exists()

    # Idempotent on re-run.
    status2 = run_nightly(store, dry_run=True, date_str="20260718",
                          artifacts_dir=art_dir, stages=["harvest"])
    assert "0 new signals" in status2["harvest"]


def test_nightly_dry_run_selected_stages_only(store, art_dir):
    status = run_nightly(store, dry_run=True, date_str="20260718",
                         artifacts_dir=art_dir, stages=["harvest", "report"])
    assert status["evolve"] == "skipped (not selected)"
    assert status["train"] == "skipped (not selected)"


# ------------------------------------------------------- end-to-end dry run

def _chat(store, user_text, reply_text, ts):
    """A chat exchange complete with the snapshot replay needs."""
    return store.record_exchange(
        user_text, reply_text, route="chat", ts=ts,
        model="Anthropic (claude-haiku-4-5-20251001)",
        context_snapshot={"system_prompt": "persona", "messages": [
            {"role": "user", "content": user_text}]},
    )


def test_dry_run_produces_real_output_end_to_end(store, art_dir, tmp_path):
    """The regression harness for the whole loop.

    Before this, `nightly --dry-run` printed "no chat exchanges with snapshots /
    nothing to judge / nothing to report" and wrote nothing at all. Now it walks
    harvest -> replay -> eval -> report with fake generation and FakeJudge, and
    leaves eval rows, a CSV, a markdown digest and a full event trail.
    """
    import time as _time

    now = _time.time()
    for i in range(3):
        _chat(store, f"question {i}", f"Answer {i}, sir.", now - 3600 + i)

    status = run_nightly(store, dry_run=True, date_str="20260814",
                         artifacts_dir=art_dir, results_dir=tmp_path / "results",
                         stages=["harvest", "replay", "eval", "report"])

    assert not any(v.startswith("FAILED") for v in status.values()), status

    # Replay generated candidates for base + facts on every exchange.
    shadow = store.query("SELECT arm, mode FROM shadow_responses")
    assert shadow, "replay produced no candidates"
    assert {r["mode"] for r in shadow} == {"replay"}
    assert "base" in {r["arm"] for r in shadow}

    # Eval judged them and wrote rows.
    verdicts = store.query("SELECT arm, prompt_id, winner FROM eval_results")
    assert verdicts, "eval judged nothing"
    assert all(v["winner"] in ("arm", "base", "tie") for v in verdicts)

    # Report wrote both artifacts.
    csv_path = tmp_path / "results" / "eval.csv"
    md_path = tmp_path / "results" / "nightly" / "20260814.md"
    assert csv_path.exists(), "no eval.csv written"
    assert md_path.exists(), "no markdown digest written"
    assert "Split: replay" in md_path.read_text()

    # And the whole run left a trail.
    events = store.events_for_run(_last_run_id(store))
    names = {e["event"] for e in events}
    assert {"run.started", "run.stage_ok", "run.finished"} <= names
    assert "replay.generated" in names
    assert "judge.verdict" in names
    assert "report.written" in names


def _last_run_id(store):
    return store.query("SELECT id FROM runs ORDER BY id DESC LIMIT 1")[0]["id"]


def test_dry_run_emits_only_documented_events(store, art_dir, tmp_path):
    """Guard on the taxonomy: an undocumented name breaks tooling silently."""
    import time as _time

    from research.events import KNOWN_EVENTS, STAGES, SUBJECT_TYPES

    now = _time.time()
    for i in range(3):
        _chat(store, f"question {i}", f"Answer {i}, sir.", now - 3600 + i)

    run_nightly(store, dry_run=True, date_str="20260814", artifacts_dir=art_dir,
                results_dir=tmp_path / "results",
                stages=["harvest", "replay", "eval", "report"])

    for row in store.query("SELECT * FROM events"):
        assert row["event"] in KNOWN_EVENTS, f"undocumented event {row['event']!r}"
        assert row["stage"] in STAGES, f"unknown stage {row['stage']!r}"
        assert row["subject_type"] in SUBJECT_TYPES


def test_replay_is_capped_and_says_so(store, art_dir, monkeypatch):
    """An uncapped replay grows without bound as the corpus does."""
    import time as _time

    from research import nightly

    monkeypatch.setattr(nightly, "REPLAY_MAX_EXCHANGES", 2)
    now = _time.time()
    for i in range(5):
        _chat(store, f"question {i}", f"Answer {i}, sir.", now - 3600 + i)

    status = run_nightly(store, dry_run=True, date_str="20260814",
                         artifacts_dir=art_dir, stages=["replay"])

    assert "3 older exchange(s) not replayed" in status["replay"]
    replayed = {r["exchange_id"] for r in
                store.query("SELECT DISTINCT exchange_id FROM shadow_responses")}
    assert len(replayed) == 2, "cap not applied"


def test_run_lifecycle_events_record_a_stage_failure(store, art_dir, monkeypatch):
    """A failing stage is isolated, and the trail says which one and why."""
    from research import nightly

    def boom(ctx):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(nightly, "STAGES", [("harvest", boom),
                                            ("replay", nightly.stage_replay)])

    status = run_nightly(store, dry_run=True, date_str="20260814",
                         artifacts_dir=art_dir)

    assert status["harvest"].startswith("FAILED: RuntimeError: disk on fire")
    assert status["replay"].startswith("ok"), "later stages still run"

    events = store.events_for_run(_last_run_id(store))
    failed = [e for e in events if e["event"] == "run.stage_failed"]
    assert len(failed) == 1
    assert json.loads(failed[0]["detail"])["stage"] == "harvest"
    assert "disk on fire" in json.loads(failed[0]["detail"])["message"]
    assert json.loads(
        [e for e in events if e["event"] == "run.finished"][0]["detail"]
    )["failed"] == ["harvest"]


# --------------------------------------------- prospective temporal split (W2)

def test_stage_order_is_prospective():
    """Replay/eval must run before any learner so artifacts are judged on
    exchanges they never saw; report/publish come last."""
    from research.nightly import STAGES

    order = [name for name, _ in STAGES]
    assert order == ["harvest", "replay", "eval", "reflect", "evolve", "train",
                     "report", "publish"]


def test_replay_excludes_exchanges_seen_by_learners(store, art_dir):
    import time as _time

    now = _time.time()
    _chat(store, "seen by trainer", "Reply, sir.", now - 3000)
    _chat(store, "also seen", "Reply, sir.", now - 2500)
    # A learner consumed everything up to now-2000 (e.g. last night's train).
    store.set_run_context(None, "train")
    store.emit("dataset.built", subject_type="dataset", subject_id="d1",
               detail={"ts_marker": True})
    store.execute("UPDATE events SET ts = ? WHERE event = 'dataset.built'",
                  (now - 2000,))
    store.set_run_context(None, "live")
    unseen = _chat(store, "fresh question", "Reply, sir.", now - 1000)

    status = run_nightly(store, dry_run=True, date_str="20260901",
                         artifacts_dir=art_dir, stages=["replay"])
    assert "cutoff" in status["replay"]
    replayed = {r["exchange_id"] for r in
                store.query("SELECT DISTINCT exchange_id FROM shadow_responses")}
    assert replayed == {unseen}


def test_report_stamps_versions_captured_at_replay_time(store, art_dir, tmp_path):
    """After the reorder, `current` pointers advance between eval and report;
    CSV rows must carry the version that was actually judged."""
    import csv as _csv
    import time as _time

    from research import artifacts, nightly

    v1 = artifacts.new_version("prompt", "20260830", art_dir)
    (v1 / "block.md").write_text("- tea, not coffee")
    artifacts.advance_current("prompt", v1.name, art_dir)

    now = _time.time()
    _chat(store, "question", "Answer, sir.", now - 600)

    # A fake learner stage advances the pointer after eval, before report.
    def fake_train(ctx):
        v2 = artifacts.new_version("prompt", "20260831", art_dir)
        (v2 / "block.md").write_text("- changed")
        artifacts.advance_current("prompt", v2.name, art_dir)
        return "advanced"

    original = nightly.STAGES
    patched = [(n, fake_train if n == "train" else fn) for n, fn in original]
    try:
        nightly.STAGES = patched
        status = run_nightly(store, dry_run=True, date_str="20260831",
                             artifacts_dir=art_dir, weekly=False,
                             results_dir=tmp_path / "results",
                             stages=["replay", "eval", "train", "report"])
    finally:
        nightly.STAGES = original
    assert not any(v.startswith("FAILED") for v in status.values()), status

    with open(tmp_path / "results" / "eval.csv", newline="") as f:
        rows = [r for r in _csv.DictReader(f) if r["arm"] == "prompt"]
    assert rows and all(r["artifact_version"] == "v20260830" for r in rows)
    assert artifacts.current_version("prompt", art_dir) == "v20260831"


def test_facts_context_hash_emitted(store, art_dir):
    import time as _time

    _chat(store, "question", "Answer, sir.", _time.time() - 600)
    run_nightly(store, dry_run=True, date_str="20260901",
                artifacts_dir=art_dir, stages=["replay"])
    events = store.query("SELECT arm, detail FROM events WHERE event = 'replay.context'")
    assert events and events[0]["arm"] == "facts"
    detail = json.loads(events[0]["detail"])
    assert set(detail) == {"block_sha256", "n_lines", "empty"}
    assert len(detail["block_sha256"]) == 64


# ---------------------------------------------------- audit + provenance (W3/W4)

def test_dry_run_records_audit_and_row_provenance(store, art_dir, tmp_path):
    import csv as _csv
    import time as _time

    now = _time.time()
    for i in range(5):
        _chat(store, f"question {i}", f"Answer {i}, sir.", now - 3600 + i)

    status = run_nightly(store, dry_run=True, date_str="20260901",
                         artifacts_dir=art_dir, weekly=False,
                         results_dir=tmp_path / "results",
                         stages=["harvest", "replay", "eval", "report"])
    assert not any(v.startswith("FAILED") for v in status.values()), status

    with open(tmp_path / "results" / "eval.csv", newline="") as f:
        rows = list(_csv.DictReader(f))
    assert rows
    for r in rows:
        assert r["rubric"].startswith("r1:")
        assert r["judge_temperature"] == "0"
        assert r["notes"] == "prospective"
        assert r["local_agreement"] != ""      # scripted FakeJudge auditor ran
        assert int(r["n_audited"]) >= 0

    # Audit verdicts persisted under the auditor's judge name (fake in dry-run),
    # and primary verdicts carry per-row provenance in scores.
    judges = {r["judge"] for r in store.query("SELECT judge FROM eval_results")}
    assert judges == {"fake"}
    scored = store.query("SELECT scores FROM eval_results WHERE scores IS NOT NULL")
    meta = json.loads(scored[0]["scores"])
    assert meta["rubric"].startswith("r1:") and meta["judge_temperature"] == 0.0
