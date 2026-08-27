"""Provider snapshots: correct on real fixtures, silent and file-free on empty paths."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from introspection import Paths, Probes, gather_status
from research.db import ResearchStore

NOW = time.time()


def make_probes(**overrides):
    defaults = dict(
        launchctl=lambda args: SimpleNamespace(
            returncode=0, stdout="123\t0\tcom.friday.nightly\n"),
        http_get=lambda url, timeout: 200,
        now=lambda: NOW,
    )
    defaults.update(overrides)
    return Probes(**defaults)


@pytest.fixture
def state(tmp_path):
    """A populated ~/.friday twin: research.db, artifacts tree, audit.db, logs."""
    state_dir = tmp_path / "friday-state"
    art_dir = state_dir / "research"
    (art_dir / "logs").mkdir(parents=True)
    (art_dir / "logs" / "live.log").write_text("log line\n")

    store = ResearchStore(str(state_dir / "research.db"))
    store.execute(
        "INSERT INTO runs (started_ts, finished_ts, stage_status) VALUES (?, ?, ?)",
        (NOW - 30 * 3600, NOW - 30 * 3600 + 200,
         json.dumps({"harvest": "ok (1s)", "train": "ok (200s)"})))
    store.execute(
        "INSERT INTO runs (started_ts, finished_ts, stage_status) VALUES (?, ?, ?)",
        (NOW - 2 * 3600, NOW - 2 * 3600 + 180,
         json.dumps({"harvest": "ok (1s)",
                     "train": "FAILED: RuntimeError: out of memory"})))
    store.execute(
        "INSERT INTO exchanges (ts, user_text, reply_text, route) VALUES (?, ?, ?, ?)",
        (NOW - 3600, "u", "r", "chat"))
    store.emit("artifact.advanced", subject_type="artifact",
               subject_id="lora/v20260802", arm="lora",
               artifact_version="lora/v20260802", detail={"previous": "v20260801"})
    store.close()

    lora = art_dir / "lora"
    (lora / "v20260801").mkdir(parents=True)
    (lora / "v20260802").mkdir()
    (lora / "current").write_text("v20260802\n")
    (lora / "v20260802" / "GATED").write_text("style_score 0.4 < 0.6\n")
    (lora / "v20260802" / "train.log").write_text(
        "\n".join(f"iter {i}" for i in range(40)))
    (lora / "v20260802" / "provenance.json").write_text(
        json.dumps({"inputs": {}, "dataset": {"n_train": 40}, "params": {"iters": 160}}))

    from core.harness.audit import AuditLog
    audit = AuditLog(str(state_dir / "audit.db"))
    audit.record(session_id="s1", workflow="self_repair", action_kind="self_repair",
                 plan_hash="abc", attempt=0, event="EXEC_OK", summary="")

    return Paths(state_dir=state_dir, research_db=state_dir / "research.db",
                 artifacts_dir=art_dir, audit_db=state_dir / "audit.db")


def test_empty_paths_report_unavailable_and_create_nothing(tmp_path):
    root = tmp_path / "nothing-here"
    paths = Paths(state_dir=root, research_db=root / "research.db",
                  artifacts_dir=root / "research", audit_db=root / "audit.db")
    status = gather_status(paths, make_probes(launchctl=lambda args: None))
    assert status["nightly"] == {"available": False}
    assert status["arms"] == {"available": False}
    assert status["activity"]["available"] is False
    assert status["storage"] == {"available": False}
    # The critical invariant: introspection must never create state
    # (ephemeral --chat/--test modes rely on it).
    assert not root.exists()


def test_nightly_snapshot_reports_failed_stage(state):
    snap = gather_status(state, make_probes())["nightly"]
    assert snap["available"] is True
    assert snap["last_run_failed_stages"] == ["train"]
    assert snap["last_run_hours_ago"] == pytest.approx(2.0, abs=0.2)
    # Both runs visible, newest first.
    assert [r["run_id"] for r in snap["runs"]] == [2, 1]
    assert snap["runs"][1]["failed_stages"] == []


def test_arms_snapshot_discovers_arm_and_gated_current(state):
    snap = gather_status(state, make_probes())["arms"]
    assert snap["available"] is True
    lora = snap["arms"]["lora"]
    assert lora["current"] == "v20260802"
    assert lora["versions"] == ["v20260801", "v20260802"]
    assert lora["gated"] is True
    assert "style_score" in lora["gated_note"]
    assert lora["dataset"] == {"n_train": 40}
    assert len(lora["train_log_tail"]) == 15
    assert lora["train_log_tail"][-1] == "iter 39"


def test_activity_snapshot_counts_audit_and_routes(state):
    snap = gather_status(state, make_probes())["activity"]
    assert snap["available"] is True
    assert snap["routes_24h"] == {"chat": 1}
    actions = snap["gated_actions"]
    assert actions and actions[0]["workflow"] == "self_repair"
    assert actions[0]["event"] == "EXEC_OK"
    assert snap["event_counts_24h"].get("artifact.advanced") == 1


def test_jobs_snapshot_uses_injected_launchctl(state):
    calls = []

    def fake_launchctl(args):
        calls.append(args)
        return SimpleNamespace(returncode=0,
                               stdout="1\t0\tcom.friday.nightly\n"
                                      "2\t0\tcom.other.job\n")

    snap = gather_status(state, make_probes(launchctl=fake_launchctl))["jobs"]
    assert calls == [["list"]]
    assert snap["launchd"] == ["com.friday.nightly"]


def test_storage_snapshot_discovers_dbs_and_logs(state):
    snap = gather_status(state, make_probes())["storage"]
    assert snap["available"] is True
    assert set(snap["databases"]) == {"research.db", "audit.db"}
    assert "live.log" in snap["logs"]
    assert snap["logs"]["live.log"]["bytes"] > 0


def test_snapshots_carry_no_user_text(state):
    """The text-free invariant: exchange texts must never surface."""
    blob = json.dumps(gather_status(state, make_probes()), default=str)
    assert '"u"' not in blob and '"r"' not in blob
    assert "user_text" not in blob and "reply_text" not in blob
