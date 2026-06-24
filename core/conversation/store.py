"""
Session persistence.

`SqliteSessionStore` is durable so long-running sessions survive a restart;
`InMemorySessionStore` is used in ephemeral mode and tests. Live, non-serializable
objects (e.g. an open browser page) must live only in `scratch` while the process
is up — on reload the owning workflow rebuilds them from durable `slots`.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .session import ACTIVE_STATUSES, Session, SessionStatus

logger = logging.getLogger(__name__)


class SessionStore(ABC):
    @abstractmethod
    def get(self, session_id: str) -> Optional[Session]: ...

    @abstractmethod
    def get_active_for_user(self, user_id: str) -> Optional[Session]: ...

    @abstractmethod
    def list_waiting(self) -> List[Session]: ...

    @abstractmethod
    def list_active_dialogue(self) -> List[Session]: ...

    @abstractmethod
    def save(self, session: Session) -> None: ...

    @abstractmethod
    def delete(self, session_id: str) -> None: ...


class InMemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def get_active_for_user(self, user_id: str) -> Optional[Session]:
        with self._lock:
            for s in self._sessions.values():
                if s.user_id == user_id and s.status in ACTIVE_STATUSES:
                    return s
            return None

    def list_waiting(self) -> List[Session]:
        with self._lock:
            return [s for s in self._sessions.values() if s.status == SessionStatus.WAITING]

    def list_active_dialogue(self) -> List[Session]:
        with self._lock:
            return [s for s in self._sessions.values() if s.status in ACTIVE_STATUSES]

    def save(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


_ACTIVE_VALUES = tuple(s.value for s in ACTIVE_STATUSES)


def _dumps(obj) -> str:
    """JSON-encode slots/scratch; drop anything not serializable rather than crash."""
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return json.dumps({k: v for k, v in obj.items()
                           if _is_jsonable(v)}) if isinstance(obj, dict) else "{}"


def _is_jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


class SqliteSessionStore(SessionStore):
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()  # one connection shared across the main + background threads
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id    TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                workflow_name TEXT NOT NULL,
                fsm_state     TEXT NOT NULL,
                slots         TEXT NOT NULL,
                scratch       TEXT NOT NULL,
                status        TEXT NOT NULL,
                timeout_s     INTEGER NOT NULL,
                wake_at       REAL,
                created_at    REAL NOT NULL,
                updated_at    REAL NOT NULL,
                expires_at    REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_status ON sessions(user_id, status)"
        )
        self._conn.commit()
        try:
            os.chmod(path, 0o600)  # session DB holds PII; keep it owner-only
        except OSError:
            pass

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        return Session(
            session_id=row["session_id"],
            user_id=row["user_id"],
            workflow_name=row["workflow_name"],
            fsm_state=row["fsm_state"],
            slots=json.loads(row["slots"]),
            scratch=json.loads(row["scratch"]),
            status=SessionStatus(row["status"]),
            timeout_s=row["timeout_s"],
            wake_at=row["wake_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
        )

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
        return self._row_to_session(row) if row else None

    def get_active_for_user(self, user_id: str) -> Optional[Session]:
        placeholders = ",".join("?" for _ in _ACTIVE_VALUES)
        with self._lock:
            cur = self._conn.execute(
                f"SELECT * FROM sessions WHERE user_id = ? AND status IN ({placeholders}) "
                "ORDER BY updated_at DESC LIMIT 1",
                (user_id, *_ACTIVE_VALUES),
            )
            row = cur.fetchone()
        return self._row_to_session(row) if row else None

    def list_waiting(self) -> List[Session]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM sessions WHERE status = ?", (SessionStatus.WAITING.value,)
            )
            rows = cur.fetchall()
        return [self._row_to_session(r) for r in rows]

    def list_active_dialogue(self) -> List[Session]:
        placeholders = ",".join("?" for _ in _ACTIVE_VALUES)
        with self._lock:
            cur = self._conn.execute(
                f"SELECT * FROM sessions WHERE status IN ({placeholders})", _ACTIVE_VALUES
            )
            rows = cur.fetchall()
        return [self._row_to_session(r) for r in rows]

    def save(self, session: Session) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions
                    (session_id, user_id, workflow_name, fsm_state, slots, scratch, status,
                     timeout_s, wake_at, created_at, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    fsm_state=excluded.fsm_state, slots=excluded.slots, scratch=excluded.scratch,
                    status=excluded.status, timeout_s=excluded.timeout_s, wake_at=excluded.wake_at,
                    updated_at=excluded.updated_at, expires_at=excluded.expires_at
                """,
                (
                    session.session_id, session.user_id, session.workflow_name, session.fsm_state,
                    _dumps(session.slots), _dumps(session.scratch), session.status.value,
                    session.timeout_s, session.wake_at, session.created_at, session.updated_at,
                    session.expires_at,
                ),
            )
            self._conn.commit()

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self._conn.commit()
