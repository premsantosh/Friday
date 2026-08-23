"""
Small bookkeeping tables that live next to the LangGraph checkpoints:

  agent_threads — per-user thread epoch. `reset()` bumps the epoch instead of
                  deleting checkpoints, so "clear history" is O(1) and the old
                  thread stays inspectable.
  agent_wakes   — self-scheduled wake-ups written by the `schedule_wakeup`
                  tool and serviced by BackgroundTaskRunner (Phase 4).

Plain sqlite3 with a lock (the SqliteSessionStore idiom) rather than aiosqlite:
these are tiny synchronous reads/writes and must work from any thread/loop.
The in-memory variant backs ephemeral mode and tests.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Wake:
    wake_id: str
    user_id: str
    wake_at: float
    payload: Dict[str, Any]


class AgentStore:
    """In-memory bookkeeping (ephemeral mode, tests)."""

    def __init__(self) -> None:
        self._epochs: Dict[str, int] = {}
        self._wakes: Dict[str, Wake] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ epochs
    def get_epoch(self, user_id: str) -> int:
        with self._lock:
            return self._epochs.get(user_id, 0)

    def bump_epoch(self, user_id: str) -> int:
        with self._lock:
            self._epochs[user_id] = self._epochs.get(user_id, 0) + 1
            return self._epochs[user_id]

    # ------------------------------------------------------------- wakes
    def add_wake(self, user_id: str, wake_at: float, payload: Dict[str, Any]) -> str:
        wake_id = uuid.uuid4().hex
        with self._lock:
            self._wakes[wake_id] = Wake(wake_id, user_id, float(wake_at), dict(payload))
        return wake_id

    def due_wakes(self, now: Optional[float] = None) -> List[Wake]:
        now = time.time() if now is None else now
        with self._lock:
            return sorted((w for w in self._wakes.values() if w.wake_at <= now),
                          key=lambda w: w.wake_at)

    def list_wakes(self, user_id: Optional[str] = None) -> List[Wake]:
        with self._lock:
            return sorted((w for w in self._wakes.values()
                           if user_id is None or w.user_id == user_id),
                          key=lambda w: w.wake_at)

    def delete_wake(self, wake_id: str) -> None:
        with self._lock:
            self._wakes.pop(wake_id, None)


class SqliteAgentStore(AgentStore):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_threads (user_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_wakes (
                wake_id  TEXT PRIMARY KEY,
                user_id  TEXT NOT NULL,
                wake_at  REAL NOT NULL,
                payload  TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_wakes_at ON agent_wakes(wake_at)")
        self._conn.commit()
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def get_epoch(self, user_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT epoch FROM agent_threads WHERE user_id = ?", (user_id,)).fetchone()
        return int(row["epoch"]) if row else 0

    def bump_epoch(self, user_id: str) -> int:
        with self._lock:
            self._conn.execute(
                "INSERT INTO agent_threads(user_id, epoch) VALUES (?, 1) "
                "ON CONFLICT(user_id) DO UPDATE SET epoch = epoch + 1", (user_id,))
            self._conn.commit()
            row = self._conn.execute(
                "SELECT epoch FROM agent_threads WHERE user_id = ?", (user_id,)).fetchone()
        return int(row["epoch"])

    def add_wake(self, user_id: str, wake_at: float, payload: Dict[str, Any]) -> str:
        wake_id = uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO agent_wakes(wake_id, user_id, wake_at, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (wake_id, user_id, float(wake_at), json.dumps(payload, default=str), time.time()))
            self._conn.commit()
        return wake_id

    def _rows_to_wakes(self, rows) -> List[Wake]:
        return [Wake(r["wake_id"], r["user_id"], float(r["wake_at"]), json.loads(r["payload"]))
                for r in rows]

    def due_wakes(self, now: Optional[float] = None) -> List[Wake]:
        now = time.time() if now is None else now
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM agent_wakes WHERE wake_at <= ? ORDER BY wake_at", (now,)).fetchall()
        return self._rows_to_wakes(rows)

    def list_wakes(self, user_id: Optional[str] = None) -> List[Wake]:
        with self._lock:
            if user_id is None:
                rows = self._conn.execute("SELECT * FROM agent_wakes ORDER BY wake_at").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM agent_wakes WHERE user_id = ? ORDER BY wake_at", (user_id,)).fetchall()
        return self._rows_to_wakes(rows)

    def delete_wake(self, wake_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM agent_wakes WHERE wake_id = ?", (wake_id,))
            self._conn.commit()
