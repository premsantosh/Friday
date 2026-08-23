"""
LangGraph agent engine for Friday (opt-in via FRIDAY_AGENT_ENGINE=langgraph).

Replaces the one-shot intent classifier (Step 3 of VoiceAssistant.process_input)
with a checkpointed tool-calling loop: durable per-user chat history, native
tool calling over the existing workflow registry, human-in-the-loop
confirmations that survive restarts, and self-scheduled wake-ups. Everything in
front of Step 3 (active sessions, keyword match, intent cache) and the
reservations workflow stay on the legacy path.

Imports are guarded (same idiom as aioesphomeapi / playwright elsewhere): Friday
boots and runs on the legacy router when langgraph isn't installed.

See docs/langgraph-integration.md.
"""

from __future__ import annotations

_IMPORT_ERROR: Exception | None = None
try:
    import langgraph  # noqa: F401
    import langchain_core  # noqa: F401
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: F401  (needs aiosqlite)
except Exception as exc:  # pragma: no cover - depends on the environment
    _IMPORT_ERROR = exc


def is_available() -> bool:
    """True when langgraph + langchain-core + the sqlite checkpointer import."""
    return _IMPORT_ERROR is None


def unavailable_reason() -> str:
    return "" if _IMPORT_ERROR is None else f"{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}"


if is_available():
    from .engine import AgentEngine, EngineDeps  # noqa: E402
    __all__ = ["AgentEngine", "EngineDeps", "is_available", "unavailable_reason"]
else:  # pragma: no cover
    __all__ = ["is_available", "unavailable_reason"]
