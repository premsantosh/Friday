"""Doctor checks: pass/warn/fail logic, crash containment, spoken summary."""

from __future__ import annotations

import fcntl
import json
import time
from types import SimpleNamespace

import pytest

from introspection import (
    CheckResult,
    CheckStatus,
    Paths,
    Probes,
    StatusProvider,
    format_report,
    run_doctor,
    summarize,
)
from introspection.registry import _PROVIDERS
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


def make_state(tmp_path, *, run_age_h=2.0, train_note="ok (200s)"):
    state_dir = tmp_path / "friday-state"
    art_dir = state_dir / "research"
    art_dir.mkdir(parents=True)
    store = ResearchStore(str(state_dir / "research.db"))
    store.execute(
        "INSERT INTO runs (started_ts, finished_ts, stage_status) VALUES (?, ?, ?)",
        (NOW - run_age_h * 3600, NOW - run_age_h * 3600 + 100,
         json.dumps({"harvest": "ok (1s)", "train": train_note})))
    store.emit("run.finished", subject_type="run", subject_id=1)
    store.close()
    return Paths(state_dir=state_dir, research_db=state_dir / "research.db",
                 artifacts_dir=art_dir, audit_db=state_dir / "audit.db")


def by_name(results):
    return {r.name: r for r in results}


def test_healthy_state_passes(tmp_path):
    checks = by_name(run_doctor(make_state(tmp_path), make_probes()))
    assert checks["nightly.research_db"].status is CheckStatus.PASS
    assert checks["nightly.last_run"].status is CheckStatus.PASS
    assert checks["nightly.event_freshness"].status is CheckStatus.PASS
    assert checks["jobs.launchd"].status is CheckStatus.PASS
    assert checks["storage.research.db"].status is CheckStatus.PASS
    assert checks["storage.ollama"].status is CheckStatus.PASS


def test_failed_stage_is_fail_and_named(tmp_path):
    paths = make_state(tmp_path, train_note="FAILED: RuntimeError: boom")
    check = by_name(run_doctor(paths, make_probes()))["nightly.last_run"]
    assert check.status is CheckStatus.FAIL
    assert "train" in check.message


def test_stale_run_warns(tmp_path):
    paths = make_state(tmp_path, run_age_h=50.0)
    check = by_name(run_doctor(paths, make_probes()))["nightly.last_run"]
    assert check.status is CheckStatus.WARN


def test_lock_held_long_warns(tmp_path):
    paths = make_state(tmp_path)
    lock_path = paths.artifacts_dir / "nightly.lock"
    lock_path.write_text("")
    stale = NOW + 5 * 3600  # probe clock 5h after the lock's mtime
    with open(lock_path, "w") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check = by_name(run_doctor(paths, make_probes(now=lambda: stale)))["nightly.lock"]
    assert check.status is CheckStatus.WARN
    assert "stuck" in check.message


def test_lock_free_passes(tmp_path):
    paths = make_state(tmp_path)
    (paths.artifacts_dir / "nightly.lock").write_text("")
    check = by_name(run_doctor(paths, make_probes()))["nightly.lock"]
    assert check.status is CheckStatus.PASS


def test_missing_launchd_job_warns(tmp_path):
    probes = make_probes(launchctl=lambda args: SimpleNamespace(
        returncode=0, stdout="1\t0\tcom.other.job\n"))
    check = by_name(run_doctor(make_state(tmp_path), probes))["jobs.launchd"]
    assert check.status is CheckStatus.WARN


def test_no_launchctl_skips(tmp_path):
    probes = make_probes(launchctl=lambda args: None)
    check = by_name(run_doctor(make_state(tmp_path), probes))["jobs.launchd"]
    assert check.status is CheckStatus.SKIP


def test_ollama_down_warns(tmp_path):
    def down(url, timeout):
        raise OSError("connection refused")

    check = by_name(run_doctor(make_state(tmp_path),
                               make_probes(http_get=down)))["storage.ollama"]
    assert check.status is CheckStatus.WARN


def test_bad_artifact_pointer_fails(tmp_path):
    paths = make_state(tmp_path)
    lora = paths.artifacts_dir / "lora"
    (lora / "v20260801").mkdir(parents=True)
    (lora / "current").write_text("v-gone\n")
    check = by_name(run_doctor(paths, make_probes()))["arms.lora.pointer"]
    assert check.status is CheckStatus.FAIL
    assert "v-gone" in check.message


def test_crashing_provider_becomes_fail_row(tmp_path):
    class Broken(StatusProvider):
        name = "broken"

        def checks(self, paths, probes):
            raise RuntimeError("kaput")

    provider = Broken()
    _PROVIDERS.append(provider)
    try:
        checks = by_name(run_doctor(make_state(tmp_path), make_probes()))
    finally:
        _PROVIDERS.remove(provider)
    assert checks["broken.provider"].status is CheckStatus.FAIL
    assert "kaput" in checks["broken.provider"].message


def test_summarize_all_green_and_with_failure():
    green = [CheckResult("a", CheckStatus.PASS, "fine")]
    assert "nominal" in summarize(green)
    mixed = [CheckResult("a", CheckStatus.PASS, "fine"),
             CheckResult("nightly.last_run", CheckStatus.FAIL, "train failed")]
    spoken = summarize(mixed)
    assert "nightly.last_run" in spoken and "1 failure" in spoken


def test_format_report_counts():
    text = format_report([CheckResult("a", CheckStatus.PASS, "fine"),
                          CheckResult("b", CheckStatus.WARN, "meh")])
    assert "PASS" in text and "WARN" in text and "1 pass, 1 warn" in text


def test_cmd_doctor_exit_codes(tmp_path, monkeypatch):
    from research.cli import main as cli_main

    healthy = make_state(tmp_path / "ok")
    monkeypatch.setattr("introspection.registry.Probes", lambda: make_probes())
    assert cli_main(["--db", str(healthy.research_db),
                     "--artifacts-dir", str(healthy.artifacts_dir), "doctor"]) == 0

    broken = make_state(tmp_path / "bad", train_note="FAILED: RuntimeError: x")
    assert cli_main(["--db", str(broken.research_db),
                     "--artifacts-dir", str(broken.artifacts_dir), "doctor"]) == 1
