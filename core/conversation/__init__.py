"""
Friday multi-turn agent framework.

Layer A — ConversationContext: short-term routing continuity (turnstile-ctx).
Layer B — Sessions: durable, possibly long-running multi-turn tasks.

See docs/multi-turn-agent-spec.md.
"""

from .context import ConversationContext
from .session import (
    Session,
    SessionStatus,
    TurnControl,
    TurnResult,
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
)
from .store import SessionStore, InMemorySessionStore, SqliteSessionStore
from .manager import SessionManager, GLOBAL_ESCAPES
from .background import BackgroundTaskRunner

__all__ = [
    "ConversationContext",
    "Session",
    "SessionStatus",
    "TurnControl",
    "TurnResult",
    "ACTIVE_STATUSES",
    "TERMINAL_STATUSES",
    "SessionStore",
    "InMemorySessionStore",
    "SqliteSessionStore",
    "SessionManager",
    "GLOBAL_ESCAPES",
    "BackgroundTaskRunner",
]
