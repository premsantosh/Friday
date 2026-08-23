"""
Checkpointer provider — the three-event-loop problem.

Friday runs several daemon threads that each own a private asyncio loop
(BackgroundTaskRunner, TelegramChannel, VoicePEChannel) plus `asyncio.run()`
per voice activation and per `run_single_interaction`. aiosqlite connections
are bound to the loop that created them, so a cached AsyncSqliteSaver would
break the first time a different thread touched it.

Default: open `AsyncSqliteSaver.from_conn_string()` as an async context manager
*per invocation* over one shared SQLite file (the saver enables WAL). Provably
safe; per-loop caching is a later optimisation if it ever shows up in latency.

Ephemeral mode (--chat / --test) uses one process-wide InMemorySaver so nothing
is written to disk — mirrors the InMemorySessionStore selection in
core/assistant.py.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class CheckpointerProvider:
    def __init__(self, path: str | None):
        """`path=None` → in-memory (ephemeral). Otherwise a SQLite file path
        (~ expanded, parent dir created lazily)."""
        self.path = os.path.expanduser(path) if path else None
        self._memory: InMemorySaver | None = None if self.path else InMemorySaver()

    @property
    def ephemeral(self) -> bool:
        return self.path is None

    @asynccontextmanager
    async def open(self) -> AsyncIterator[BaseCheckpointSaver]:
        if self._memory is not None:
            yield self._memory
            return
        assert self.path is not None
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(self.path) as saver:
            # The checkpoint DB carries chat history (PII); keep it owner-only,
            # same posture as sessions.db / audit.db.
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
            yield saver
