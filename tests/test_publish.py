"""The publish stage: PII gate + pathspec-only git commit of results files."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research.publish import commit_results, pii_findings, scan_targets


# ------------------------------------------------------------------ PII scan

def _paths(tmp_path, text: str) -> list[Path]:
    p = tmp_path / "eval.csv"
    p.write_text(text)
    return [p]


def test_scan_flags_emails_phones_and_guarded_names(tmp_path):
    findings = pii_findings(_paths(tmp_path, "contact bob@example.com now"), [])
    assert findings == ["eval.csv:1 email"]

    findings = pii_findings(_paths(tmp_path, "call 415-555-1234 today"), [])
    assert findings == ["eval.csv:1 phone-like"]

    findings = pii_findings(_paths(tmp_path, "call +14155551234 today"), [])
    assert findings == ["eval.csv:1 phone-like"]

    findings = pii_findings(_paths(tmp_path, "| Bilbo | 0.500 |"), ["bilbo"])
    assert findings == ["eval.csv:1 guarded-name"]
    # Findings never contain the matched text itself.
    assert all("Bilbo" not in f and "bilbo" not in f for f in findings)


def test_scan_ignores_the_numbers_results_are_made_of(tmp_path):
    text = "\n".join([
        "20260830,12,lora,base,v20260830,sonnet:claude-sonnet-5,replay,4,4,0,4",
        "0.062,0.1250,,,0.750,1.000",
        "date 2026-08-30 run 1788119910 latency 12522ms",
        "| memory | 87.5% | 0.250 | n/a | 100%/100% |",
    ])
    assert pii_findings(_paths(tmp_path, text), []) == []


def test_guarded_names_come_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FRIDAY_PII_NAMES", "gandalf, frodo")
    from research.publish import guarded_names
    assert guarded_names() == ["gandalf", "frodo"]


# ---------------------------------------------------------------- git commit

@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.invalid"],
                   cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    results = tmp_path / "results"
    (results / "nightly").mkdir(parents=True)
    (results / "eval.csv").write_text("date,arm\n20260830,lora\n")
    (results / "nightly" / "20260830.md").write_text("# Eval report\n")
    return tmp_path, results


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def test_commit_results_commits_only_result_paths(repo):
    root, results = repo
    unrelated = root / "wip.py"
    unrelated.write_text("x = 1\n")

    note = commit_results(results, "20260830")
    assert note.startswith("committed")

    log = _git(root, "log", "--oneline").stdout
    assert "results: nightly 20260830" in log
    shown = _git(root, "show", "--name-only", "--format=").stdout
    assert "results/eval.csv" in shown and "results/nightly/20260830.md" in shown
    assert "wip.py" not in shown
    # The unrelated dirty file is untouched and uncommitted.
    assert unrelated.exists()
    assert "wip.py" in _git(root, "status", "--porcelain").stdout


def test_commit_results_noop_when_nothing_changed(repo):
    root, results = repo
    assert commit_results(results, "20260830").startswith("committed")
    assert commit_results(results, "20260830") == "nothing to commit"


def test_commit_results_outside_a_repo(tmp_path):
    results = tmp_path / "results"
    (results / "nightly").mkdir(parents=True)
    (results / "eval.csv").write_text("date\n")
    assert commit_results(results, "20260830") == "skipped: not a git repo"


def test_scan_targets_lists_csv_and_digests(repo):
    _, results = repo
    names = [p.name for p in scan_targets(results)]
    assert names == ["eval.csv", "20260830.md"]


# ------------------------------------------------------------ stage_publish

def test_stage_publish_blocks_on_pii_and_alerts(repo, monkeypatch):
    from research.db import ResearchStore
    from research.nightly import NightlyContext, stage_publish

    root, results = repo
    (results / "nightly" / "20260830.md").write_text("email me at a@b.co\n")
    alerts = []
    import introspection.alerts as alerts_mod
    monkeypatch.setattr(alerts_mod, "send_telegram", lambda text: alerts.append(text) or True)

    store = ResearchStore(str(root / "research.db"))
    ctx = NightlyContext(store=store, date_str="20260830", since_ts=0.0,
                         dry_run=True, results_dir=results)
    note = stage_publish(ctx)
    assert note.startswith("BLOCKED: 1 PII finding")
    assert alerts and "PII gate" in alerts[0]
    assert "Eval report" not in _git(root, "log", "--oneline").stdout  # no commit
    events = store.query("SELECT event FROM events")
    assert any(e["event"] == "results.blocked" for e in events)
    store.close()


def test_stage_publish_dry_run_scans_but_skips_commit(repo):
    from research.db import ResearchStore
    from research.nightly import NightlyContext, stage_publish

    root, results = repo
    store = ResearchStore(str(root / "research.db"))
    ctx = NightlyContext(store=store, date_str="20260830", since_ts=0.0,
                         dry_run=True, results_dir=results)
    assert stage_publish(ctx) == "scan clean, skipped commit (dry-run)"
    assert not _git(root, "log", "--oneline").stdout.strip()
    store.close()


def test_stage_publish_commits_for_real(repo):
    from research.db import ResearchStore
    from research.nightly import NightlyContext, stage_publish

    root, results = repo
    store = ResearchStore(str(root / "research.db"))
    ctx = NightlyContext(store=store, date_str="20260830", since_ts=0.0,
                         dry_run=False, results_dir=results)
    note = stage_publish(ctx)
    assert note.startswith("committed")
    assert any(e["event"] == "results.committed"
               for e in store.query("SELECT event FROM events"))
    store.close()
