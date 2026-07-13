import re
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Words too common to be useful in fact search.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "did",
    "do", "does", "what", "when", "where", "who", "how", "why", "you", "your",
    "me", "my", "i", "we", "it", "that", "this", "for", "of", "to", "in", "on",
    "at", "about", "with", "remember", "last", "time", "yesterday", "earlier",
    "before", "tell", "know", "said",
}


class FridayStore:
    """Persistent memory for Friday. All data stays local on your Mac Mini."""

    def __init__(self, db_path: str = "~/.friday/memory.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        # A single connection is shared between the main thread and the
        # background fact-extraction thread — serialize all access.
        self._lock = threading.Lock()
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent reads
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_tables()

    def _init_tables(self):
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    confidence REAL DEFAULT 1.0,
                    source TEXT DEFAULT 'conversation',
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now')),
                    access_count INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tokens_estimated INTEGER,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary TEXT NOT NULL,
                    turn_range_start INTEGER,
                    turn_range_end INTEGER,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
                CREATE INDEX IF NOT EXISTS idx_turns_created ON conversation_turns(created_at);
            """)
            self.conn.commit()

    # --- Facts (long-term memory) ---

    def remember(self, key: str, value: str, category: str = "general",
                 confidence: float = 1.0):
        with self._lock:
            self.conn.execute("""
                INSERT INTO facts (key, value, category, confidence, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value, category=excluded.category,
                    confidence=excluded.confidence, updated_at=datetime('now')
            """, (key, value, category, confidence))
            self.conn.commit()

    def recall(self, key: str) -> Optional[str]:
        with self._lock:
            row = self.conn.execute(
                "UPDATE facts SET access_count = access_count + 1 WHERE key = ? RETURNING value",
                (key,)
            ).fetchone()
            self.conn.commit()
            return row[0] if row else None

    def recall_by_category(self, category: str, min_confidence: float = 0.0) -> dict[str, str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT key, value FROM facts WHERE category = ? AND confidence >= ?"
                " ORDER BY access_count DESC",
                (category, min_confidence)
            ).fetchall()
            return {k: v for k, v in rows}

    def search_facts(self, query: str, min_confidence: float = 0.0,
                     limit: int = 10) -> list[tuple[str, str, str]]:
        """Keyword search across facts.

        The query is tokenized and stopwords dropped, so a whole sentence
        ("what did I say my coffee order was") still matches the fact keyed
        "coffee_order". Upgrade to vector search later.
        """
        keywords = [
            w for w in re.findall(r"[a-z0-9']+", query.lower())
            if len(w) > 2 and w not in _STOPWORDS
        ]
        if not keywords:
            return []

        clauses = " OR ".join("key LIKE ? OR value LIKE ?" for _ in keywords)
        params: list = []
        for w in keywords:
            params.extend([f"%{w}%", f"%{w}%"])
        params.extend([min_confidence, limit])

        with self._lock:
            rows = self.conn.execute(
                f"SELECT key, value, category FROM facts WHERE ({clauses})"
                f" AND confidence >= ? ORDER BY confidence DESC, access_count DESC LIMIT ?",
                params
            ).fetchall()
            return rows

    # --- Conversation history ---

    def log_turn(self, role: str, content: str):
        tokens_est = len(content) // 4  # rough estimate
        with self._lock:
            self.conn.execute(
                "INSERT INTO conversation_turns (role, content, tokens_estimated) VALUES (?, ?, ?)",
                (role, content, tokens_est)
            )
            self.conn.commit()

    def get_recent_turns(self, n: int = 10) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT role, content, created_at FROM conversation_turns ORDER BY id DESC LIMIT ?",
                (n,)
            ).fetchall()
            return [{"role": r, "content": c, "timestamp": t} for r, c, t in reversed(rows)]

    def count_turns(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0]

    def get_oldest_turns(self, n: int) -> list[dict]:
        """Oldest turns first, with their row ids — used by the summarizer so it
        can delete exactly the rows it summarized."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, role, content FROM conversation_turns ORDER BY id ASC LIMIT ?",
                (n,)
            ).fetchall()
            return [{"id": i, "role": r, "content": c} for i, r, c in rows]

    def save_summary(self, summary: str, start_id: int, end_id: int):
        with self._lock:
            self.conn.execute(
                "INSERT INTO conversation_summaries (summary, turn_range_start, turn_range_end) VALUES (?, ?, ?)",
                (summary, start_id, end_id)
            )
            self.conn.commit()

    def delete_turns(self, ids: list[int]):
        if not ids:
            return
        with self._lock:
            placeholders = ",".join("?" for _ in ids)
            self.conn.execute(
                f"DELETE FROM conversation_turns WHERE id IN ({placeholders})", ids
            )
            self.conn.commit()

    def get_summaries(self, n: int = 5) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT summary FROM conversation_summaries ORDER BY id DESC LIMIT ?",
                (n,)
            ).fetchall()
            return [r[0] for r in reversed(rows)]

    def bump_confidence(self, key: str, delta: float = 0.05):
        """Increase confidence for a fact, capped at 1.0."""
        with self._lock:
            self.conn.execute(
                "UPDATE facts SET confidence = MIN(1.0, confidence + ?), updated_at = datetime('now') WHERE key = ?",
                (delta, key),
            )
            self.conn.commit()

    def drop_confidence(self, key: str, delta: float = 0.3):
        """Decrease confidence for a fact. Delete if it falls below 0.1."""
        with self._lock:
            self.conn.execute(
                "UPDATE facts SET confidence = MAX(0.0, confidence - ?), updated_at = datetime('now') WHERE key = ?",
                (delta, key),
            )
            self.conn.execute("DELETE FROM facts WHERE key = ? AND confidence < 0.1", (key,))
            self.conn.commit()

    # --- Storage management (important with 256GB SSD) ---

    def get_db_size_mb(self) -> float:
        return self.db_path.stat().st_size / (1024 * 1024)

    def prune_old_turns(self, days: int = 30):
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with self._lock:
            self.conn.execute(
                "DELETE FROM conversation_turns WHERE created_at < ?", (cutoff,)
            )
            self.conn.execute("VACUUM")
            self.conn.commit()
