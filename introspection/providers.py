"""Core status providers: nightly runs, artifact arms, activity, jobs, storage.

Everything here is discovery-driven — arms are whatever version directories
exist, databases are whatever ``*.db`` files sit under ``~/.friday``, scheduled
jobs are whatever launchd knows under the ``com.friday.`` label — so state a
future ability leaves in the conventional places is reported with no changes
to this file. See registry.py for the invariants (read-only, text-free,
never raise).
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from introspection.registry import (
    CheckResult,
    CheckStatus,
    Paths,
    Probes,
    StatusProvider,
    register_provider,
)

LAUNCHD_LABEL_PREFIX = "com.friday."
NIGHTLY_LABEL = "com.friday.nightly"
NIGHTLY_MAX_AGE_H = 26.0        # 03:30 daily + slack
LOCK_STALE_H = 4.0              # a training run should never hold the lock this long
EVENT_STALE_H = 48.0
LOG_WARN_BYTES = 50 * 1024 * 1024
DISK_PASS_GB = 20.0             # mlx training headroom
DISK_WARN_GB = 8.0
TRAIN_LOG_TAIL_LINES = 15

# Directories under ~/.friday that are stores in their own right (ChromaDB etc.)
_KNOWN_STORE_DIRS = ("intent_cache",)


def _open_ro(path: Path) -> Optional[sqlite3.Connection]:
    """Read-only connection; None if the file doesn't exist. Never creates."""
    path = Path(path).expanduser()
    if not path.exists():
        return None
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _query(conn: sqlite3.Connection, sql: str, params=()) -> List[tuple]:
    return conn.execute(sql, params).fetchall()


def _hours_ago(ts: Optional[float], now: float) -> Optional[float]:
    if ts is None:
        return None
    return round((now - ts) / 3600.0, 1)


# ------------------------------------------------------------------- nightly

class NightlyProvider(StatusProvider):
    """The learning loop: recent runs, per-stage status, schedule, lock."""

    name = "nightly"

    def snapshot(self, paths: Paths, probes: Probes) -> Dict[str, Any]:
        conn = _open_ro(paths.research_db)
        if conn is None:
            return {"available": False}
        try:
            rows = _query(conn, "SELECT id, started_ts, finished_ts, stage_status"
                                " FROM runs ORDER BY id DESC LIMIT 3")
        finally:
            conn.close()
        now = probes.now()
        runs = []
        for run_id, started, finished, stage_status in rows:
            stages = _parse_stages(stage_status)
            runs.append({
                "run_id": run_id,
                "started_ts": started,
                "finished_ts": finished,
                "hours_ago": _hours_ago(started, now),
                "stages": stages,
                "failed_stages": [s for s, note in stages.items()
                                  if str(note).startswith("FAILED")],
            })
        return {
            "available": True,
            "runs": runs,
            "last_run_hours_ago": runs[0]["hours_ago"] if runs else None,
            "last_run_failed_stages": runs[0]["failed_stages"] if runs else [],
            "schedule": _nightly_schedule(),
            "lock": _lock_state(paths, probes),
        }

    def checks(self, paths: Paths, probes: Probes) -> List[CheckResult]:
        results = [self._check_db(paths)]
        if Path(paths.research_db).expanduser().exists():
            results.append(self._check_last_run(paths, probes))
            results.append(self._check_events_fresh(paths, probes))
        results.append(self._check_lock(paths, probes))
        results.append(_check_eval_placeholders())
        return results

    def _check_db(self, paths: Paths) -> CheckResult:
        conn = _open_ro(paths.research_db)
        if conn is None:
            return CheckResult("nightly.research_db", CheckStatus.WARN,
                               "no research.db yet (research substrate not started)")
        try:
            ok = conn.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            conn.close()
        if ok == "ok":
            return CheckResult("nightly.research_db", CheckStatus.PASS,
                               "research.db opens and passes quick_check")
        return CheckResult("nightly.research_db", CheckStatus.FAIL,
                           f"quick_check: {ok}")

    def _check_last_run(self, paths: Paths, probes: Probes) -> CheckResult:
        conn = _open_ro(paths.research_db)
        try:
            rows = _query(conn, "SELECT started_ts, stage_status FROM runs"
                                " ORDER BY id DESC LIMIT 1")
        finally:
            conn.close()
        if not rows:
            return CheckResult("nightly.last_run", CheckStatus.WARN,
                               "no nightly run recorded yet")
        started, stage_status = rows[0]
        stages = _parse_stages(stage_status)
        failed = [s for s, note in stages.items() if str(note).startswith("FAILED")]
        age_h = _hours_ago(started, probes.now())
        if failed:
            return CheckResult("nightly.last_run", CheckStatus.FAIL,
                               f"last run ({age_h}h ago) failed at: {', '.join(failed)}")
        if age_h is not None and age_h > NIGHTLY_MAX_AGE_H:
            return CheckResult("nightly.last_run", CheckStatus.WARN,
                               f"last run was {age_h}h ago (expected daily at 03:30)")
        return CheckResult("nightly.last_run", CheckStatus.PASS,
                           f"last run {age_h}h ago, all stages ok")

    def _check_events_fresh(self, paths: Paths, probes: Probes) -> CheckResult:
        conn = _open_ro(paths.research_db)
        try:
            rows = _query(conn, "SELECT MAX(ts) FROM events")
        finally:
            conn.close()
        newest = rows[0][0] if rows else None
        if newest is None:
            return CheckResult("nightly.event_freshness", CheckStatus.WARN,
                               "events table is empty")
        age_h = _hours_ago(newest, probes.now())
        if age_h is not None and age_h > EVENT_STALE_H:
            return CheckResult("nightly.event_freshness", CheckStatus.WARN,
                               f"newest event is {age_h}h old — recording may be off")
        return CheckResult("nightly.event_freshness", CheckStatus.PASS,
                           f"newest event {age_h}h old")

    def _check_lock(self, paths: Paths, probes: Probes) -> CheckResult:
        state = _lock_state(paths, probes)
        if not state["exists"]:
            return CheckResult("nightly.lock", CheckStatus.PASS, "no lock file")
        if not state["held"]:
            return CheckResult("nightly.lock", CheckStatus.PASS, "lock is free")
        age_h = state.get("age_hours")
        if age_h is not None and age_h >= LOCK_STALE_H:
            return CheckResult("nightly.lock", CheckStatus.WARN,
                               f"lock held for {age_h}h — a run may be stuck")
        return CheckResult("nightly.lock", CheckStatus.PASS,
                           "lock held (a nightly run is in progress)")


def _parse_stages(stage_status: Optional[str]) -> Dict[str, str]:
    try:
        data = json.loads(stage_status or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _lock_state(paths: Paths, probes: Probes) -> Dict[str, Any]:
    lock_path = Path(paths.artifacts_dir).expanduser() / "nightly.lock"
    if not lock_path.exists():
        return {"exists": False, "held": False}
    held = False
    try:
        with open(lock_path) as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            except BlockingIOError:
                held = True
    except OSError:
        return {"exists": True, "held": None}
    return {"exists": True, "held": held,
            "age_hours": _hours_ago(lock_path.stat().st_mtime, probes.now())}


def _nightly_schedule() -> Dict[str, Any]:
    """Discover installed com.friday.* launchd plists (macOS convention)."""
    jobs = []
    agents_dir = Path("~/Library/LaunchAgents").expanduser()
    if agents_dir.exists():
        for plist in sorted(agents_dir.glob(f"{LAUNCHD_LABEL_PREFIX}*.plist")):
            jobs.append(_read_plist_schedule(plist))
    return {"installed_jobs": jobs,
            "expected": f"{NIGHTLY_LABEL} daily at 03:30 (launchd)"}


def _read_plist_schedule(plist_path: Path) -> Dict[str, Any]:
    import plistlib

    try:
        with open(plist_path, "rb") as f:
            data = plistlib.load(f)
        cal = data.get("StartCalendarInterval") or {}
        return {"label": data.get("Label", plist_path.stem),
                "calendar": {k: v for k, v in cal.items()} if isinstance(cal, dict) else cal}
    except Exception:
        return {"label": plist_path.stem, "calendar": None}


def _check_eval_placeholders() -> CheckResult:
    try:
        from research.evalset import count_placeholders

        n = count_placeholders()
    except Exception:
        return CheckResult("nightly.eval_placeholders", CheckStatus.SKIP,
                           "curated eval set not readable")
    if n:
        return CheckResult("nightly.eval_placeholders", CheckStatus.WARN,
                           f"{n} curated probe(s) still marked FILL-IN")
    return CheckResult("nightly.eval_placeholders", CheckStatus.PASS,
                       "no placeholder probes")


# ---------------------------------------------------------------------- arms

class ArmsProvider(StatusProvider):
    """Model/artifact arms, discovered from the artifacts directory itself."""

    name = "arms"

    def snapshot(self, paths: Paths, probes: Probes) -> Dict[str, Any]:
        art_dir = Path(paths.artifacts_dir).expanduser()
        if not art_dir.exists():
            return {"available": False}
        arms: Dict[str, Any] = {}
        for arm in discover_arms(art_dir):
            arms[arm] = _arm_state(arm, art_dir)
        return {"available": True, "arms": arms}

    def checks(self, paths: Paths, probes: Probes) -> List[CheckResult]:
        art_dir = Path(paths.artifacts_dir).expanduser()
        if not art_dir.exists():
            return [CheckResult("arms.artifacts_dir", CheckStatus.SKIP,
                                "no artifacts directory yet")]
        results = []
        for arm in discover_arms(art_dir):
            state = _arm_state(arm, art_dir)
            current = state["current"]
            if current is None:
                results.append(CheckResult(f"arms.{arm}.pointer", CheckStatus.PASS,
                                           "no current version (never advanced)"))
            elif not state["current_exists"]:
                results.append(CheckResult(f"arms.{arm}.pointer", CheckStatus.FAIL,
                                           f"current points at missing version {current!r}"))
            elif state["gated"]:
                results.append(CheckResult(f"arms.{arm}.pointer", CheckStatus.WARN,
                                           f"current version {current} is GATED"))
            else:
                results.append(CheckResult(f"arms.{arm}.pointer", CheckStatus.PASS,
                                           f"current={current} "
                                           f"({state['version_count']} version(s))"))
        return results


def discover_arms(artifacts_dir: Path) -> List[str]:
    """An arm is any subdirectory holding a `current` pointer or v* version dirs."""
    arms = []
    for entry in sorted(artifacts_dir.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "current").exists() or any(
                p.is_dir() and p.name.startswith("v") for p in entry.iterdir()):
            arms.append(entry.name)
    return arms


def _arm_state(arm: str, art_dir: Path) -> Dict[str, Any]:
    from research import artifacts, provenance

    current = artifacts.current_version(arm, art_dir)
    versions = artifacts.list_versions(arm, art_dir)
    state: Dict[str, Any] = {
        "current": current,
        "current_exists": bool(current) and (art_dir / arm / current).exists(),
        "versions": versions,
        "version_count": len(versions),
        "gated": False,
    }
    if current and state["current_exists"]:
        version_dir = art_dir / arm / current
        gated = version_dir / "GATED"
        if gated.exists():
            state["gated"] = True
            state["gated_note"] = gated.read_text().strip()[:200]
        manifest = provenance.read_manifest(arm, current, art_dir)
        if manifest:
            state["dataset"] = manifest.get("dataset")
            state["params"] = manifest.get("params")
        train_log = version_dir / "train.log"
        if train_log.exists():
            lines = train_log.read_text(errors="replace").splitlines()
            state["train_log_tail"] = lines[-TRAIN_LOG_TAIL_LINES:]
    return state


# ------------------------------------------------------------------ activity

class ActivityProvider(StatusProvider):
    """What Friday actually did: gated actions (audit log) and routed traffic."""

    name = "activity"

    def snapshot(self, paths: Paths, probes: Probes) -> Dict[str, Any]:
        now = probes.now()
        cutoff = now - 24 * 3600
        out: Dict[str, Any] = {"available": False, "window_hours": 24}

        conn = _open_ro(paths.audit_db)
        if conn is not None:
            try:
                rows = _query(conn,
                              "SELECT ts, workflow, action_kind, event, summary"
                              " FROM audit_events WHERE ts >= ?"
                              " ORDER BY id DESC LIMIT 50", (cutoff,))
            finally:
                conn.close()
            out["available"] = True
            out["gated_actions"] = [
                {"ts": ts, "workflow": wf, "action_kind": kind, "event": event,
                 "summary": summary or ""}
                for ts, wf, kind, event, summary in rows
            ]

        conn = _open_ro(paths.research_db)
        if conn is not None:
            try:
                routes = _query(conn, "SELECT COALESCE(route, 'unknown'), COUNT(*)"
                                      " FROM exchanges WHERE ts >= ?"
                                      " GROUP BY 1", (cutoff,))
                events = _query(conn, "SELECT event, COUNT(*) FROM events"
                                      " WHERE ts >= ? GROUP BY 1"
                                      " ORDER BY 2 DESC", (cutoff,))
            finally:
                conn.close()
            out["available"] = True
            out["routes_24h"] = dict(routes)
            out["event_counts_24h"] = dict(events)
        return out


# ---------------------------------------------------------------------- jobs

class JobsProvider(StatusProvider):
    """Scheduled and pending work: launchd jobs, agent wakes, live sessions."""

    name = "jobs"

    def snapshot(self, paths: Paths, probes: Probes) -> Dict[str, Any]:
        out: Dict[str, Any] = {"available": True}
        out["launchd"] = _launchd_jobs(probes)

        conn = _open_ro(Path(paths.state_dir) / "agent_checkpoints.db")
        if conn is not None:
            try:
                rows = _query(conn, "SELECT wake_id, user_id, wake_at FROM agent_wakes"
                                    " ORDER BY wake_at LIMIT 20")
            except sqlite3.Error:
                rows = []
            finally:
                conn.close()
            now = probes.now()
            out["pending_wakes"] = [
                {"wake_id": wid, "user_id": uid, "wake_at": at,
                 "due_in_hours": round((at - now) / 3600, 1)}
                for wid, uid, at in rows
            ]

        conn = _open_ro(Path(paths.state_dir) / "sessions.db")
        if conn is not None:
            try:
                rows = _query(conn, "SELECT workflow_name, status, COUNT(*)"
                                    " FROM sessions WHERE status IN"
                                    " ('active', 'awaiting_confirmation', 'waiting')"
                                    " GROUP BY 1, 2")
            except sqlite3.Error:
                rows = []
            finally:
                conn.close()
            out["live_sessions"] = [
                {"workflow": wf, "status": status, "count": n}
                for wf, status, n in rows
            ]
        return out

    def checks(self, paths: Paths, probes: Probes) -> List[CheckResult]:
        jobs = _launchd_jobs(probes)
        if jobs is None:
            return [CheckResult("jobs.launchd", CheckStatus.SKIP,
                                "launchctl not available on this platform")]
        if NIGHTLY_LABEL in jobs:
            return [CheckResult("jobs.launchd", CheckStatus.PASS,
                                f"{NIGHTLY_LABEL} is loaded")]
        return [CheckResult("jobs.launchd", CheckStatus.WARN,
                            f"{NIGHTLY_LABEL} is not loaded "
                            f"(run research/scripts/install_nightly.sh)")]


def _launchd_jobs(probes: Probes) -> Optional[List[str]]:
    """Labels of loaded com.friday.* launchd jobs; None when unsupported."""
    try:
        proc = probes.launchctl(["list"])
    except Exception:
        return None
    if proc is None:
        return None
    labels = []
    for line in (proc.stdout or "").splitlines():
        label = line.split("\t")[-1].strip()
        if label.startswith(LAUNCHD_LABEL_PREFIX):
            labels.append(label)
    return labels


# ------------------------------------------------------------------- storage

class StorageProvider(StatusProvider):
    """Databases, logs and disk — discovered by globbing the state directory."""

    name = "storage"

    def snapshot(self, paths: Paths, probes: Probes) -> Dict[str, Any]:
        state_dir = Path(paths.state_dir).expanduser()
        if not state_dir.exists():
            return {"available": False}
        out: Dict[str, Any] = {"available": True, "databases": {}, "logs": {},
                               "stores": {}}
        for db in sorted(state_dir.glob("*.db")):
            out["databases"][db.name] = {"bytes": db.stat().st_size}
        for store_name in _KNOWN_STORE_DIRS:
            store = state_dir / store_name
            if store.is_dir():
                out["stores"][store_name] = {"entries": sum(1 for _ in store.iterdir())}
        logs_dir = Path(paths.artifacts_dir).expanduser() / "logs"
        if logs_dir.exists():
            now = probes.now()
            for log in sorted(logs_dir.iterdir()):
                if log.is_file():
                    out["logs"][log.name] = {
                        "bytes": log.stat().st_size,
                        "age_hours": _hours_ago(log.stat().st_mtime, now),
                    }
        try:
            usage = shutil.disk_usage(state_dir)
            out["disk_free_gb"] = round(usage.free / 1e9, 1)
        except OSError:
            pass
        return out

    def checks(self, paths: Paths, probes: Probes) -> List[CheckResult]:
        results: List[CheckResult] = []
        state_dir = Path(paths.state_dir).expanduser()
        if not state_dir.exists():
            return [CheckResult("storage.state_dir", CheckStatus.SKIP,
                                "no ~/.friday state directory yet")]
        for db in sorted(state_dir.glob("*.db")):
            conn = _open_ro(db)
            if conn is None:
                continue
            try:
                conn.execute("SELECT 1").fetchone()
                results.append(CheckResult(f"storage.{db.name}", CheckStatus.PASS,
                                           f"openable ({db.stat().st_size // 1024} KB)"))
            except sqlite3.Error as exc:
                results.append(CheckResult(f"storage.{db.name}", CheckStatus.FAIL,
                                           f"cannot read: {exc}"))
            finally:
                conn.close()
        try:
            free_gb = shutil.disk_usage(state_dir).free / 1e9
            if free_gb >= DISK_PASS_GB:
                status = CheckStatus.PASS
            elif free_gb >= DISK_WARN_GB:
                status = CheckStatus.WARN
            else:
                status = CheckStatus.FAIL
            results.append(CheckResult("storage.disk_free", status,
                                       f"{free_gb:.1f} GB free "
                                       f"(training wants ≥{DISK_PASS_GB:.0f} GB)"))
        except OSError:
            results.append(CheckResult("storage.disk_free", CheckStatus.SKIP,
                                       "disk usage unavailable"))
        logs_dir = Path(paths.artifacts_dir).expanduser() / "logs"
        if logs_dir.exists():
            for log in sorted(logs_dir.iterdir()):
                if log.is_file() and log.stat().st_size > LOG_WARN_BYTES:
                    results.append(CheckResult(
                        f"storage.log.{log.name}", CheckStatus.WARN,
                        f"{log.stat().st_size // (1024 * 1024)} MB and unrotated"))
        results.append(self._check_ollama(probes))
        return results

    def _check_ollama(self, probes: Probes) -> CheckResult:
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        try:
            code = probes.http_get(f"{base}/api/tags", 2.0)
        except Exception:
            return CheckResult("storage.ollama", CheckStatus.WARN,
                               "Ollama unreachable (shadow/local models degrade)")
        if code == 200:
            return CheckResult("storage.ollama", CheckStatus.PASS, "Ollama responds")
        return CheckResult("storage.ollama", CheckStatus.WARN,
                           f"Ollama answered HTTP {code}")


register_provider(NightlyProvider())
register_provider(ArmsProvider())
register_provider(ActivityProvider())
register_provider(JobsProvider())
register_provider(StorageProvider())
