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
