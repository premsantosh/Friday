"""Append-only research store (~/.friday/research.db).

Separate from production memory.db on purpose: memory.db prunes raw turns after
summarization, while the study needs a permanent, replayable record. Nothing in
this schema is ever deleted — corrections are modeled as new rows (feedback,
retired flags), not updates in place.

Every write path here also appends a provenance row to `events` inside the same
transaction, so nothing can enter the study untraced (see research/events.py).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS exchanges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    channel TEXT,
    user_id TEXT,
    user_text TEXT NOT NULL,
    reply_text TEXT NOT NULL,
    route TEXT,
    latency_ms INTEGER,
    model TEXT,
    context_snapshot TEXT,
    memory_turn_id INTEGER
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange_id INTEGER NOT NULL REFERENCES exchanges(id),
    kind TEXT NOT NULL,
    signal INTEGER NOT NULL,
    source TEXT NOT NULL,
    ts REAL NOT NULL,
    details TEXT
);

CREATE TABLE IF NOT EXISTS shadow_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange_id INTEGER NOT NULL REFERENCES exchanges(id),
    arm TEXT NOT NULL,
    mode TEXT NOT NULL,
    model_tag TEXT,
    artifact_version TEXT,
    response_text TEXT,
    gen_ms INTEGER,
    ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_ts REAL NOT NULL,
    finished_ts REAL,
    stage_status TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    arm TEXT NOT NULL,
    prompt_id TEXT,
    opponent TEXT,
    winner TEXT,
    judge TEXT,
    scores TEXT,
    ts REAL NOT NULL
);

-- Curated-split generations, persisted so the human anchor can re-present the
-- exact pairs the judge saw and so a weekly eval is reproducible after the
-- fact. (Replay generations live in shadow_responses; curated prompt ids are
-- not exchange ids, hence a separate table.)
CREATE TABLE IF NOT EXISTS curated_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    prompt_id TEXT NOT NULL,
    arm TEXT NOT NULL,
    artifact_version TEXT,
    response_text TEXT,
    ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    importance REAL,
    source_exchange_ids TEXT,
    retired INTEGER NOT NULL DEFAULT 0
);

-- Provenance timeline: what the loop used to improve itself, and when. The
-- subject is the thing you would trace (usually the exchange), not the row that
-- was written; detail carries ids/counts/reasons but never user or reply text.
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    run_id INTEGER,
    stage TEXT NOT NULL,
    event TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    arm TEXT,
    artifact_version TEXT,
    detail TEXT
);

CREATE INDEX IF NOT EXISTS idx_exchanges_ts ON exchanges(ts);
CREATE INDEX IF NOT EXISTS idx_feedback_exchange ON feedback(exchange_id);
CREATE INDEX IF NOT EXISTS idx_shadow_exchange ON shadow_responses(exchange_id);
CREATE INDEX IF NOT EXISTS idx_events_subject ON events(subject_type, subject_id, id);
CREATE INDEX IF NOT EXISTS idx_events_artifact ON events(artifact_version, id);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event, ts);

-- Feedback invariant: one row per (exchange, source). Early builds let the
-- Telegram buttons insert a row per tap; collapse any such duplicates to the
-- newest before enforcing (idempotent, runs on every open).
DELETE FROM feedback WHERE id NOT IN
    (SELECT MAX(id) FROM feedback GROUP BY exchange_id, source);
CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_unique
    ON feedback(exchange_id, source);
"""

_EXCHANGE_UPDATABLE = {"channel", "user_id", "route", "latency_ms"}


class ResearchStore:
    """Thread-safe sqlite wrapper. Same WAL idioms as FridayStore."""

    def __init__(self, db_path: str = "~/.friday/research.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        # Tagged onto every event; nightly sets these per stage, live leaves them.
        self._run_id: Optional[int] = None
        self._stage: str = "live"
        # Emit is deliberately unable to raise, so surface breakage as a counter
        # rather than silence (an invisible failure is how the loop starved).
        self.emit_failures = 0
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    # ------------------------------------------------------------ generic SQL
    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Locked read returning plain dicts (rows never outlive the lock)."""
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def execute(self, sql: str, params: tuple = ()) -> int:
        """Locked write + commit; returns lastrowid."""
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return int(cur.lastrowid)

    def backup_to(self, path: Path) -> None:
        """Locked sqlite online backup to `path`."""
        dest = sqlite3.connect(str(path))
        try:
            with self._lock, dest:
                self.conn.backup(dest)
        finally:
            dest.close()

    # ---------------------------------------------------------------- events
    def set_run_context(self, run_id: Optional[int], stage: str) -> None:
        """Tag subsequent events with the nightly run/stage that caused them.

        The live substrate leaves the default (None, 'live'). Nightly runs in a
        separate process under flock, so plain attributes are safe here; if the
        orchestrator ever runs in-process beside the recorder, make these
        threading.local.
        """
        self._run_id = run_id
        self._stage = stage

    def _emit_locked(
        self,
        event: str,
        subject_type: str,
        subject_id: Any,
        *,
        arm: Optional[str] = None,
        artifact_version: Optional[str] = None,
        detail: Optional[dict] = None,
        ts: Optional[float] = None,
    ) -> None:
        """Append an event row on an already-held lock, without committing.

        Callers inside an existing transaction use this so the event and the
        write it describes land together or not at all.
        """
        self.conn.execute(
            "INSERT INTO events (ts, run_id, stage, event, subject_type, subject_id,"
            " arm, artifact_version, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts if ts is not None else time.time(), self._run_id, self._stage, event,
             subject_type, str(subject_id), arm, artifact_version,
             json.dumps(detail) if detail else None),
        )

    def emit(
        self,
        event: str,
        *,
        subject_type: str,
        subject_id: Any,
        arm: Optional[str] = None,
        artifact_version: Optional[str] = None,
        detail: Optional[dict] = None,
        ts: Optional[float] = None,
    ) -> None:
        """Append one provenance event.

        Never raises: the loop's bookkeeping must not be able to break a reply
        or abort a nightly stage. Failures bump emit_failures and log a warning,
        both surfaced by `python -m research status`.
        """
        try:
            with self._lock:
                self._emit_locked(event, subject_type, subject_id, arm=arm,
                                  artifact_version=artifact_version, detail=detail,
                                  ts=ts)
                self.conn.commit()
        except Exception:
            self.emit_failures += 1
            logger.warning("Event emit failed: %s (%s %s)", event, subject_type,
                           subject_id, exc_info=True)

    def emit_all(
        self,
        event: str,
        *,
        subject_type: str,
        subject_ids: Iterable[Any],
        arm: Optional[str] = None,
        artifact_version: Optional[str] = None,
        detail: Optional[dict] = None,
    ) -> None:
        """One row per subject, same event and detail, in a single transaction."""
        try:
            with self._lock:
                for subject_id in subject_ids:
                    self._emit_locked(event, subject_type, subject_id, arm=arm,
                                      artifact_version=artifact_version,
                                      detail=detail)
                self.conn.commit()
        except Exception:
            self.emit_failures += 1
            logger.warning("Batch event emit failed: %s (%s)", event, subject_type,
                           exc_info=True)

    def events_for(self, subject_type: str, subject_id: Any) -> list[dict]:
        return self.query(
            "SELECT * FROM events WHERE subject_type = ? AND subject_id = ? ORDER BY id",
            (subject_type, str(subject_id)),
        )

    def events_for_artifact(self, artifact_version: str) -> list[dict]:
        """Everything that fed, or came from, one artifact version."""
        return self.query(
            "SELECT * FROM events WHERE artifact_version = ?"
            " OR (subject_type = 'artifact' AND subject_id = ?) ORDER BY id",
            (artifact_version, artifact_version),
        )

    def events_for_run(self, run_id: int) -> list[dict]:
        return self.query("SELECT * FROM events WHERE run_id = ? ORDER BY id", (run_id,))

    def recent_events(self, limit: int = 20, *, event: Optional[str] = None,
                      arm: Optional[str] = None, since_ts: float = 0.0) -> list[dict]:
        sql = "SELECT * FROM events WHERE ts >= ?"
        params: list[Any] = [since_ts]
        if event:
            sql += " AND event = ?"
            params.append(event)
        if arm:
            sql += " AND arm = ?"
            params.append(arm)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return self.query(sql, tuple(params))

    # ------------------------------------------------------------- exchanges
    def record_exchange(
        self,
        user_text: str,
        reply_text: str,
        *,
        route: Optional[str] = None,
        channel: Optional[str] = None,
        user_id: Optional[str] = None,
        latency_ms: Optional[int] = None,
        model: Optional[str] = None,
        context_snapshot: Optional[dict] = None,
        memory_turn_id: Optional[int] = None,
        ts: Optional[float] = None,
    ) -> int:
        snapshot_json = json.dumps(context_snapshot) if context_snapshot is not None else None
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO exchanges (ts, channel, user_id, user_text, reply_text, route,"
                " latency_ms, model, context_snapshot, memory_turn_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts if ts is not None else time.time(), channel, user_id, user_text,
                 reply_text, route, latency_ms, model, snapshot_json, memory_turn_id),
            )
            exchange_id = int(cur.lastrowid)
            # has_snapshot is the field that makes a starved ingress obvious:
            # no snapshot means no shadow, no replay, no eval for this turn.
            self._emit_locked(
                "exchange.recorded", "exchange", exchange_id,
                detail={"route": route, "channel": channel, "user_id": user_id,
                        "model": model, "has_snapshot": snapshot_json is not None,
                        "latency_ms": latency_ms,
                        "memory_turn_id": memory_turn_id},
            )
            self.conn.commit()
            return exchange_id

    def update_exchange(self, exchange_id: int, **fields: Any) -> None:
        """Backfill routing metadata on an existing exchange (never content)."""
        unknown = set(fields) - _EXCHANGE_UPDATABLE
        if unknown:
            raise ValueError(f"not backfillable: {sorted(unknown)}")
        fields = {k: v for k, v in fields.items() if v is not None}
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self.conn.execute(
                f"UPDATE exchanges SET {sets} WHERE id = ?",
                (*fields.values(), exchange_id),
            )
            self._emit_locked("exchange.backfilled", "exchange", exchange_id,
                              detail=dict(fields))
            self.conn.commit()

    def get_exchange(self, exchange_id: int) -> Optional[dict]:
        # Reads take the lock too: one sqlite3 connection shared between the
        # shadow worker thread and the caller's thread segfaults without full
        # serialization (observed, not theoretical).
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM exchanges WHERE id = ?", (exchange_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("context_snapshot"):
            d["context_snapshot"] = json.loads(d["context_snapshot"])
        return d

    def exchanges_since(self, ts: float) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, ts, channel, user_id, user_text, reply_text, route, latency_ms"
                " FROM exchanges WHERE ts >= ? ORDER BY ts",
                (ts,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------- feedback
    def add_feedback(
        self,
        exchange_id: int,
        kind: str,
        signal: int,
        source: str,
        details: Optional[str] = None,
    ) -> int:
        """First-wins insert (miners): a later row for the same (exchange,
        source) is silently ignored. Returns the row id, or 0 when ignored."""
        with self._lock:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO feedback (exchange_id, kind, signal, source, ts, details)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (exchange_id, kind, signal, source, time.time(), details),
            )
            feedback_id = int(cur.lastrowid) if cur.rowcount else 0
            self._emit_locked(
                "feedback.added" if feedback_id else "feedback.duplicate_ignored",
                "exchange", exchange_id,
                detail={"feedback_id": feedback_id, "kind": kind, "signal": signal,
                        "source": source, "details": details},
            )
            self.conn.commit()
            return feedback_id

    def upsert_feedback(
        self,
        exchange_id: int,
        kind: str,
        signal: int,
        source: str,
        details: Optional[str] = None,
    ) -> None:
        """Latest-wins upsert (explicit 👍/👎 buttons): re-pressing the same
        button is a no-op in effect, pressing the other one replaces the
        earlier choice. The single deliberate exception to append-only —
        exactly one button row per exchange, atomic under the unique index."""
        with self._lock:
            prior = self.conn.execute(
                "SELECT signal FROM feedback WHERE exchange_id = ? AND source = ?",
                (exchange_id, source),
            ).fetchone()
            self.conn.execute(
                "INSERT INTO feedback (exchange_id, kind, signal, source, ts, details)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(exchange_id, source) DO UPDATE SET"
                " kind = excluded.kind, signal = excluded.signal,"
                " ts = excluded.ts, details = excluded.details",
                (exchange_id, kind, signal, source, time.time(), details),
            )
            self._emit_locked(
                "feedback.upserted", "exchange", exchange_id,
                detail={"kind": kind, "signal": signal, "source": source,
                        "replaced": prior["signal"] if prior is not None else None},
            )
            self.conn.commit()

    def feedback_for(self, exchange_id: int) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM feedback WHERE exchange_id = ? ORDER BY ts", (exchange_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def has_feedback(self, exchange_id: int, source: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM feedback WHERE exchange_id = ? AND source = ? LIMIT 1",
                (exchange_id, source),
            ).fetchone()
        return row is not None

    # ---------------------------------------------------------------- shadow
    def add_shadow_response(
        self,
        exchange_id: int,
        arm: str,
        mode: str,
        response_text: str,
        *,
        model_tag: Optional[str] = None,
        artifact_version: Optional[str] = None,
        gen_ms: Optional[int] = None,
    ) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO shadow_responses (exchange_id, arm, mode, model_tag,"
                " artifact_version, response_text, gen_ms, ts)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (exchange_id, arm, mode, model_tag, artifact_version, response_text,
                 gen_ms, time.time()),
            )
            row_id = int(cur.lastrowid)
            self._emit_locked(
                "replay.generated" if mode == "replay" else "shadow.generated",
                "exchange", exchange_id, arm=arm, artifact_version=artifact_version,
                detail={"shadow_response_id": row_id, "model_tag": model_tag,
                        "gen_ms": gen_ms, "chars": len(response_text or "")},
            )
            self.conn.commit()
            return row_id

    def add_curated_response(
        self,
        run_id: Optional[int],
        prompt_id: str,
        arm: str,
        response_text: str,
        *,
        artifact_version: Optional[str] = None,
    ) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO curated_responses (run_id, prompt_id, arm,"
                " artifact_version, response_text, ts) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, prompt_id, arm, artifact_version, response_text,
                 time.time()),
            )
            row_id = int(cur.lastrowid)
            self._emit_locked(
                "eval.candidate_recorded", "prompt", prompt_id, arm=arm,
                artifact_version=artifact_version,
                detail={"curated_response_id": row_id,
                        "chars": len(response_text or "")},
            )
            self.conn.commit()
            return row_id

    # ---------------------------------------------------------------- status
    def counts(self) -> dict[str, int]:
        out = {}
        with self._lock:
            for table in ("exchanges", "feedback", "shadow_responses", "runs",
                          "eval_results", "memories", "events"):
                out[table] = self.conn.execute(
                    f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return out

    def close(self) -> None:
        with self._lock:
            self.conn.close()
