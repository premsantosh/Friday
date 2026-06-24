"""
Append-only audit log for gated actions (harness spec §4).

Every gate decision and every execution attempt of an irreversible action is
recorded as an *event row* — rows are never updated or deleted, so the log is
the authoritative record of what Friday actually did. It also backs the gate's
idempotency policy: `last_event_for()` tells the gate whether an identical
action already started or succeeded, so a replayed turn can never double-fire.

The `summary` column is built by the gate from a whitelist of plan fields —
never the whole plan, never card data.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Event types (append-only; a (session, kind, plan_hash, attempt) thread reads
# bottom-up: GATE_DENY*, then EXEC_STARTED → EXEC_OK | EXEC_FAIL).
GATE_DENY = "GATE_DENY"
EXEC_STARTED = "EXEC_STARTED"
EXEC_OK = "EXEC_OK"
EXEC_FAIL = "EXEC_FAIL"

DEFAULT_PATH = "~/.friday/audit.db"


class AuditLog:
    """SQLite-backed, append-only. Connects lazily so merely constructing a
    workflow doesn't create files; thread-safe like SqliteSessionStore."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = os.path.expanduser(path)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    @classmethod
    def from_env(cls) -> "AuditLog":
        return cls(os.getenv("FRIDAY_AUDIT_DB", DEFAULT_PATH))

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.path != ":memory:":
                parent = os.path.dirname(os.path.abspath(self.path))
                os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          REAL NOT NULL,
                    session_id  TEXT NOT NULL,
                    workflow    TEXT NOT NULL,
                    action_kind TEXT NOT NULL,
                    plan_hash   TEXT NOT NULL,
                    attempt     INTEGER NOT NULL DEFAULT 0,
                    event       TEXT NOT NULL,
                    policy      TEXT,
                    code        TEXT,
                    summary     TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_action "
                "ON audit_events(session_id, action_kind, plan_hash)"
            )
            conn.commit()
            if self.path != ":memory:":
                try:
                    os.chmod(self.path, 0o600)  # the log names businesses/dates; owner-only
                except OSError:
                    pass
            self._conn = conn
        return self._conn

    def record(self, *, session_id: str, workflow: str, action_kind: str,
               plan_hash: str, event: str, attempt: int = 0,
               policy: Optional[str] = None, code: Optional[str] = None,
               summary: str = "") -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO audit_events "
                "(ts, session_id, workflow, action_kind, plan_hash, attempt, event, policy, code, summary) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), session_id, workflow, action_kind, plan_hash,
                 attempt, event, policy, code, summary),
            )
            conn.commit()

    def last_event_for(self, session_id: str, action_kind: str, plan_hash: str,
                       attempt: int = 0) -> Optional[Dict[str, Any]]:
        """Latest *execution* event (not gate denials) for this exact action."""
        with self._lock:
            conn = self._connect()
            cur = conn.execute(
                "SELECT * FROM audit_events "
                "WHERE session_id = ? AND action_kind = ? AND plan_hash = ? AND attempt = ? "
                "AND event IN (?, ?, ?) ORDER BY id DESC LIMIT 1",
                (session_id, action_kind, plan_hash, attempt,
                 EXEC_STARTED, EXEC_OK, EXEC_FAIL),
            )
            row = cur.fetchone()
        return dict(row) if row else None
