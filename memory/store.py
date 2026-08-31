import sqlite3
import json
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Kept in sync with research/approaches/facts_baseline.py, which must keep
# its own copy so the study arm stays self-describing.
_SEARCH_STOPWORDS = {
    "what", "when", "where", "which", "about", "should", "could", "would",
    "have", "this", "that", "with", "from", "make", "give", "tell", "please",
}
MAX_SEARCH_FACTS = 6


class FridayStore:
    """Persistent memory for Friday. All data stays local on your Mac Mini."""
    
    def __init__(self, db_path: str = "~/.friday/memory.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent reads
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_tables()
    
    def _init_tables(self):
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
        # The extractor's LLM sometimes emits non-string values; sqlite can't
        # bind them and the whole extraction thread used to die on it.
        key = str(key).strip()
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            value = json.dumps(value)
        elif not isinstance(value, str):
            value = str(value)
        self.conn.execute("""
            INSERT INTO facts (key, value, category, confidence, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE SET 
                value=excluded.value, category=excluded.category,
                confidence=excluded.confidence, updated_at=datetime('now')
        """, (key, value, category, confidence))
        self.conn.commit()
    
    def recall(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            "UPDATE facts SET access_count = access_count + 1 WHERE key = ? RETURNING value",
            (key,)
        ).fetchone()
        self.conn.commit()
        return row[0] if row else None
    
    def recall_by_category(self, category: str, min_confidence: float = 0.5) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT key, value FROM facts WHERE category = ? AND confidence >= ?"
            " ORDER BY access_count DESC",
            (category, min_confidence)
        ).fetchall()
        return {k: v for k, v in rows}

    def search_facts(self, query: str, min_confidence: float = 0.5) -> list[tuple[str, str, str]]:
        """Keyword search across facts. Splits the query into content words —
        a whole-sentence LIKE pattern never matches anything."""
        words = [w for w in re.findall(r"[a-zA-Z]{4,}", query.lower())
                 if w not in _SEARCH_STOPWORDS]
        seen: dict[str, tuple[str, str, str]] = {}
        for word in words:
            rows = self.conn.execute(
                "SELECT key, value, category FROM facts"
                " WHERE (key LIKE ? OR value LIKE ?) AND confidence >= ?",
                (f"%{word}%", f"%{word}%", min_confidence)
            ).fetchall()
            for key, value, category in rows:
                seen.setdefault(key, (key, value, category))
        return list(seen.values())[:MAX_SEARCH_FACTS]
    
    # --- Conversation history ---
    
    def log_turn(self, role: str, content: str) -> int:
        tokens_est = len(content) // 4  # rough estimate
        cur = self.conn.execute(
            "INSERT INTO conversation_turns (role, content, tokens_estimated) VALUES (?, ?, ?)",
            (role, content, tokens_est)
        )
        self.conn.commit()
        return int(cur.lastrowid)
    
    def get_recent_turns(self, n: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content, created_at FROM conversation_turns ORDER BY id DESC LIMIT ?",
            (n,)
        ).fetchall()
        return [{"role": r, "content": c, "timestamp": t} for r, c, t in reversed(rows)]
    
    def save_summary(self, summary: str, start_id: int, end_id: int):
        self.conn.execute(
            "INSERT INTO conversation_summaries (summary, turn_range_start, turn_range_end) VALUES (?, ?, ?)",
            (summary, start_id, end_id)
        )
        # Prune the summarized raw turns to save disk
        self.conn.execute(
            "DELETE FROM conversation_turns WHERE id BETWEEN ? AND ?",
            (start_id, end_id)
        )
        self.conn.commit()
    
    def get_summaries(self, n: int = 5) -> list[str]:
        rows = self.conn.execute(
            "SELECT summary FROM conversation_summaries ORDER BY id DESC LIMIT ?",
            (n,)
        ).fetchall()
        return [r[0] for r in reversed(rows)]
    
    def bump_confidence(self, key: str, delta: float = 0.05):
        """Increase confidence for a fact, capped at 1.0."""
        self.conn.execute(
            "UPDATE facts SET confidence = MIN(1.0, confidence + ?), updated_at = datetime('now') WHERE key = ?",
            (delta, key),
        )
        self.conn.commit()

    def drop_confidence(self, key: str, delta: float = 0.3):
        """Decrease confidence for a fact. Delete if it falls below 0.1."""
        self.conn.execute(
            "UPDATE facts SET confidence = MAX(0.0, confidence - ?), updated_at = datetime('now') WHERE key = ?",
            (delta, key),
        )
        self.conn.execute("DELETE FROM facts WHERE confidence < 0.1")
        self.conn.commit()

    # --- Storage management (important with 256GB SSD) ---
    
    def get_db_size_mb(self) -> float:
        return self.db_path.stat().st_size / (1024 * 1024)
    
    def prune_old_turns(self, days: int = 30):
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        self.conn.execute(
            "DELETE FROM conversation_turns WHERE created_at < ?", (cutoff,)
        )
        self.conn.execute("VACUUM")
        self.conn.commit()
