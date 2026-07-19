"""Append-only research store (~/.friday/research.db).

Separate from production memory.db on purpose: memory.db prunes raw turns after
summarization, while the study needs a permanent, replayable record. Nothing in
this schema is ever deleted — corrections are modeled as new rows (feedback,
retired flags), not updates in place.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

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

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    importance REAL,
    source_exchange_ids TEXT,
    retired INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_exchanges_ts ON exchanges(ts);
CREATE INDEX IF NOT EXISTS idx_feedback_exchange ON feedback(exchange_id);
CREATE INDEX IF NOT EXISTS idx_shadow_exchange ON shadow_responses(exchange_id);
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
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

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
            self.conn.commit()
            return int(cur.lastrowid)

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
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO feedback (exchange_id, kind, signal, source, ts, details)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (exchange_id, kind, signal, source, time.time(), details),
            )
            self.conn.commit()
            return int(cur.lastrowid)

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
            self.conn.commit()
            return int(cur.lastrowid)

    # ---------------------------------------------------------------- status
    def counts(self) -> dict[str, int]:
        out = {}
        with self._lock:
            for table in ("exchanges", "feedback", "shadow_responses", "runs",
                          "eval_results", "memories"):
                out[table] = self.conn.execute(
                    f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return out

    def close(self) -> None:
        with self._lock:
            self.conn.close()
