"""
Layer B — BackgroundTaskRunner (MT3).

A single daemon thread (modeled on the coffee-machine monitor) that drives
long-running work for the SessionManager:

  1. sweep_expired()  — close abandoned ACTIVE/AWAITING dialogue sessions.
  2. tick_waiting()   — call on_tick() for WAITING sessions whose wake_at has passed
                        (poll availability, pick up an async call/email outcome…).
  3. notify           — when a tick produces a user-facing result (it completed,
                        needs confirmation, or was cancelled), push the message out
                        via the supplied notifier (the assistant's speak callback by
                        default; a workflow can layer Signal/SMS on top).

Notifications are suppressed for results that simply keep waiting (BACKGROUND)
or carry no message.

  4. agent wake-ups (optional `agent_engine`) — the LangGraph engine's
     self-scheduled follow-ups ("re-check at 6pm"): due `agent_wakes` rows are
     serviced by re-invoking that user's thread and the reply goes out through
     the same notifier. Without an engine the runner behaves exactly as before.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable, Optional

from .session import TurnControl

logger = logging.getLogger(__name__)

# Controls whose result should be announced to the user when produced by a tick.
_NOTIFY_CONTROLS = (TurnControl.COMPLETE, TurnControl.AWAIT_CONFIRMATION, TurnControl.CANCEL)


class BackgroundTaskRunner:
    def __init__(
        self,
        session_manager,
        notify: Callable[[str], None],
        tick_seconds: int = 30,
        agent_engine=None,
    ):
        self.sessions = session_manager
        self._notify = notify
        self.tick_seconds = max(1, int(tick_seconds))
        self.agent_engine = agent_engine   # optional agent.AgentEngine (run_due_wakes)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="session-runner")
        self._thread.start()
        logger.info("BackgroundTaskRunner started (tick=%ss).", self.tick_seconds)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    # ------------------------------------------------------------------ internals
    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_loop())
        finally:
            loop.close()

    async def _async_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("Session runner tick failed — will retry next interval.")
            for _ in range(self.tick_seconds):
                if self._stop_event.is_set():
                    return
                await asyncio.sleep(1)

    async def _tick(self) -> None:
        if self.sessions is not None:
            self.sessions.sweep_expired()
            for session, result in await self.sessions.tick_waiting():
                if result.message and result.control in _NOTIFY_CONTROLS:
                    try:
                        self._notify(result.message)
                    except Exception:
                        logger.exception("Notifier failed for session %s", session.session_id)

        if self.agent_engine is not None:
            try:
                replies = await self.agent_engine.run_due_wakes()
            except Exception:
                logger.exception("Agent wake-ups failed — will retry next interval.")
                replies = []
            for message in replies:
                try:
                    self._notify(message)
                except Exception:
                    logger.exception("Notifier failed for an agent wake-up")
