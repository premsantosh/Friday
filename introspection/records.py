"""Read access to Friday's historical records: runs, evals, artifacts, feedback,
and the owner's own conversation transcript.

This module backs the `evals` / `history` / `insights` topics of `self_status`
and the `recall_conversation` workflow. It is NOT a set of StatusProviders:
providers (providers.py) stay text-free, while conversation recall necessarily
returns the owner's own words. The carve-out is deliberate and bounded —
everything here is still strictly read-only (`_open_ro`, never creates state,
safe in ephemeral modes), the data never leaves the machine except through
Friday's spoken reply to its owner, and excerpt lengths are capped so a tool
message stays small.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from introspection.providers import _open_ro, _parse_stages

EXCERPT_CHARS = 200
REPORT_CHARS = 4000
SERIES_CAP = 30           # most callers want trends, not a full dump


# ------------------------------------------------------------------ evals

def eval_history(csv_path: Optional[Path] = None) -> Dict[str, Any]:
    """The longitudinal eval record (results/eval.csv): win-rate series per
    (arm, split) plus where each arm stands against the pre-registered bar."""
    from research import protocol

    try:
        rows = protocol.load_rows(csv_path)
    except FileNotFoundError:
        return {"available": False}

    rows = [r for r in rows if r.get("judge") not in protocol.EXCLUDED_JUDGES]
    series: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = f"{row.get('arm')}/{row.get('split')}"
        series.setdefault(key, []).append({
            "date": row.get("date"),
            "win_rate": _float(row.get("win_rate")),
            "p_value": _float(row.get("p_value")),
            "n_prompts": _int(row.get("n_prompts")),
            "artifact_version": row.get("artifact_version") or None,
            "judge": row.get("judge"),
        })
    for key in series:
        series[key].sort(key=lambda e: _row_date(e["date"]) or datetime.min)
        series[key] = series[key][-SERIES_CAP:]

    arms = sorted({r.get("arm") for r in rows if r.get("arm")})
    bars = {}
    for arm in arms:
        bar = protocol.evaluate_bar(rows, arm=arm, split="curated")
        bars[arm] = {
            "improved": bar.improved,
            "summary": bar.summary(),
            "conditions": [{"number": c.number, "name": c.name,
                            "passed": c.passed, "detail": c.detail}
                           for c in bar.conditions],
        }
    return {"available": True, "series": series, "bars": bars,
            "eval_dates": sorted({r.get("date") for r in rows})}


def nightly_reports(results_dir: Optional[Path] = None,
                    limit: int = 5) -> Dict[str, Any]:
    """The newest nightly markdown digests (code-generated aggregates; they
    carry the per-category win rates the CSV flattens away)."""
    from research.report import RESULTS_DIR

    reports_dir = (Path(results_dir) if results_dir else RESULTS_DIR) / "nightly"
    if not reports_dir.exists():
        return {"available": False}
    files = sorted(reports_dir.glob("*.md"))[-limit:]
    return {"available": True,
            "reports": [{"name": f.name,
                         "text": f.read_text(errors="replace")[:REPORT_CHARS]}
                        for f in files]}


# ------------------------------------------------------------------- runs

def runs_history(db_path: Path, limit: int = SERIES_CAP) -> Dict[str, Any]:
    """Every recorded nightly run (newest first) with per-stage outcomes,
    plus reliability aggregates."""
    conn = _open_ro(db_path)
    if conn is None:
        return {"available": False}
    try:
        rows = conn.execute(
            "SELECT id, started_ts, finished_ts, stage_status FROM runs"
            " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    finally:
        conn.close()

    runs = []
    stage_failures: Dict[str, int] = {}
    for run_id, started, finished, stage_status in rows:
        stages = _parse_stages(stage_status)
        failed = [s for s, note in stages.items() if str(note).startswith("FAILED")]
        for stage in failed:
            stage_failures[stage] = stage_failures.get(stage, 0) + 1
        runs.append({
            "run_id": run_id,
            "date": _day(started),
            "duration_s": round(finished - started) if finished and started else None,
            "stages": stages,
            "failed_stages": failed,
        })
    n_failed = sum(1 for r in runs if r["failed_stages"])
    return {
        "available": True,
        "runs": runs,
        "total": len(runs),
        "runs_with_failures": n_failed,
        "most_common_failing_stage": (max(stage_failures, key=stage_failures.get)
                                      if stage_failures else None),
    }


# -------------------------------------------------------------- artifacts

def artifact_history(arm: str, artifacts_dir: Path) -> Dict[str, Any]:
    """Every version of an arm with its provenance manifest — the dataset
    growth and parameter series — plus gate outcomes."""
    from research import artifacts, provenance

    art_dir = Path(artifacts_dir).expanduser()
    versions = artifacts.list_versions(arm, art_dir)
    if not versions:
        return {"available": False, "arm": arm}
    current = artifacts.current_version(arm, art_dir)
    history = []
    for version in versions[-SERIES_CAP:]:
        manifest = provenance.read_manifest(arm, version, art_dir) or {}
        entry: Dict[str, Any] = {
            "version": version,
            "is_current": version == current,
            "gated": (art_dir / arm / version / "GATED").exists(),
            "dataset": manifest.get("dataset"),
            "params": manifest.get("params"),
            "inputs": manifest.get("inputs"),
        }
        history.append(entry)
    return {"available": True, "arm": arm, "current": current,
            "versions": history}


# --------------------------------------------------------------- feedback

def feedback_stats(db_path: Path, days: int = 30, *,
                   now: Optional[float] = None) -> Dict[str, Any]:
    """Feedback signal counts by source and by day. Signals and sources only —
    never the underlying text."""
    import time as _time

    conn = _open_ro(db_path)
    if conn is None:
        return {"available": False}
    cutoff = (now or _time.time()) - days * 86400
    try:
        by_source = conn.execute(
            "SELECT source, signal, COUNT(*) FROM feedback WHERE ts >= ?"
            " GROUP BY source, signal", (cutoff,)).fetchall()
        by_day = conn.execute(
            "SELECT date(ts, 'unixepoch'), SUM(signal > 0), SUM(signal < 0)"
            " FROM feedback WHERE ts >= ? GROUP BY 1 ORDER BY 1", (cutoff,)).fetchall()
    finally:
        conn.close()
    positives = sum(n for _, sig, n in by_source if sig > 0)
    negatives = sum(n for _, sig, n in by_source if sig < 0)
    return {
        "available": True,
        "window_days": days,
        "positive": positives,
        "negative": negatives,
        "by_source": [{"source": s, "signal": sig, "count": n}
                      for s, sig, n in by_source],
        "by_day": [{"day": d, "positive": p or 0, "negative": m or 0}
                   for d, p, m in by_day],
    }


# ----------------------------------------------------------- conversations

def conversation_search(db_path: Path, *, query: Optional[str] = None,
                        since_ts: Optional[float] = None,
                        until_ts: Optional[float] = None,
                        limit: int = 20,
                        memory_db: Optional[Path] = None) -> Dict[str, Any]:
    """Search the permanent transcript (research.db `exchanges`) by date window
    and/or keyword. Excerpts are truncated to EXCERPT_CHARS. When research.db
    is absent, falls back to dated summaries from memory.db."""
    conn = _open_ro(db_path)
    if conn is None:
        return _summary_fallback(memory_db, query=query, since_ts=since_ts,
                                 until_ts=until_ts)
    where, params = ["1=1"], []
    if since_ts is not None:
        where.append("ts >= ?")
        params.append(since_ts)
    if until_ts is not None:
        where.append("ts < ?")
        params.append(until_ts)
    if query:
        where.append("(LOWER(user_text) LIKE ? OR LOWER(reply_text) LIKE ?)")
        needle = f"%{query.lower()}%"
        params.extend([needle, needle])
    try:
        rows = conn.execute(
            f"SELECT ts, channel, route, user_text, reply_text FROM exchanges"
            f" WHERE {' AND '.join(where)} ORDER BY ts DESC LIMIT ?",
            (*params, limit)).fetchall()
        total = conn.execute(
            f"SELECT COUNT(*) FROM exchanges WHERE {' AND '.join(where)}",
            tuple(params)).fetchone()[0]
    finally:
        conn.close()
    matches = [{
        "ts": ts,
        "day": _day(ts),
        "channel": channel,
        "route": route,
        "user": (user or "")[:EXCERPT_CHARS],
        "reply": (reply or "")[:EXCERPT_CHARS],
    } for ts, channel, route, user, reply in rows]
    matches.reverse()  # oldest first reads naturally
    return {"available": True, "source": "exchanges", "total_matches": total,
            "matches": matches}


def _summary_fallback(memory_db: Optional[Path], *, query: Optional[str],
                      since_ts: Optional[float],
                      until_ts: Optional[float]) -> Dict[str, Any]:
    if memory_db is None:
        return {"available": False}
    conn = _open_ro(memory_db)
    if conn is None:
        return {"available": False}
    try:
        rows = conn.execute(
            "SELECT summary, created_at FROM conversation_summaries"
            " ORDER BY id DESC LIMIT 20").fetchall()
    except Exception:
        return {"available": False}
    finally:
        conn.close()
    summaries = []
    for summary, created_at in rows:
        ts = _iso_to_ts(created_at)
        if since_ts is not None and (ts is None or ts < since_ts):
            continue
        if until_ts is not None and (ts is None or ts >= until_ts):
            continue
        if query and query.lower() not in summary.lower():
            continue
        summaries.append({"day": (created_at or "")[:10],
                          "summary": summary[:EXCERPT_CHARS * 2]})
    summaries.reverse()
    return {"available": True, "source": "summaries",
            "total_matches": len(summaries), "matches": summaries}


# ------------------------------------------------------------ date parsing

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday")


def past_date(text: str, now: datetime) -> Optional[Tuple[float, float]]:
    """Resolve a spoken reference to a PAST day (or short window) into a
    (start_ts, end_ts) epoch pair. The harness's normalize_date is the
    opposite tool — booking-oriented, future-biased, rejects the past — so
    recall gets its own resolver. Returns None when nothing date-like parses.
    """
    if not text or not text.strip():
        return None
    s = text.strip().lower()
    today = now.date()

    if "today" in s:
        return _day_window(today)
    if "yesterday" in s:
        return _day_window(today - timedelta(days=1))
    if "last week" in s:
        start = today - timedelta(days=today.weekday() + 7)
        return _window(start, start + timedelta(days=7))
    if "this week" in s:
        return _window(today - timedelta(days=today.weekday()),
                       today + timedelta(days=1))
    for idx, name in enumerate(_WEEKDAYS):
        if name in s:
            back = (today.weekday() - idx) % 7 or 7   # most recent past, never today
            return _day_window(today - timedelta(days=back))
    try:
        return _day_window(date.fromisoformat(s))
    except ValueError:
        pass
    try:
        from dateutil import parser as date_parser

        parsed = date_parser.parse(s, default=now, fuzzy=True).date()
    except Exception:
        return None
    if parsed > today:  # "June 1" after June 1 means the one that passed
        parsed = date(parsed.year - 1, parsed.month, parsed.day)
    return _day_window(parsed)


def _day_window(day: date) -> Tuple[float, float]:
    return _window(day, day + timedelta(days=1))


def _window(start: date, end: date) -> Tuple[float, float]:
    to_ts = lambda d: datetime(d.year, d.month, d.day).timestamp()  # noqa: E731
    return to_ts(start), to_ts(end)


# ---------------------------------------------------------------- helpers

def _float(raw: Any) -> Optional[float]:
    try:
        return float(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _int(raw: Any) -> Optional[int]:
    value = _float(raw)
    return int(value) if value is not None else None


def _row_date(raw: Any) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(str(raw or "").strip(), fmt)
        except ValueError:
            continue
    return None


def _day(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _iso_to_ts(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None
