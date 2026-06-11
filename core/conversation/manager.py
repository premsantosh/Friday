"""
SessionManager — lifecycle and turn routing for multi-turn task sessions.

At most one ACTIVE dialogue session per user; many WAITING sessions may coexist.
The manager applies each `TurnResult` (merge slots, set fsm_state, transition
status, persist) and keeps the Layer-A context register aligned to the active
workflow so follow-ups bias toward the in-progress task.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .session import Session, SessionStatus, TurnControl, TurnResult

logger = logging.getLogger(__name__)

# Words that always abort the active session, regardless of workflow.
GLOBAL_ESCAPES = {"cancel", "stop", "never mind", "nevermind", "forget it", "abort", "scrap that"}


class SessionManager:
    def __init__(self, store, workflows, default_timeout_s: int = 600, context=None):
        self.store = store
        self.workflows = workflows          # WorkflowManager
        self.default_timeout_s = default_timeout_s
        self.context = context              # optional ConversationContext

    # ------------------------------------------------------------------ queries
    def get_active(self, user_id: str) -> Optional[Session]:
        session = self.store.get_active_for_user(user_id)
        if session is None:
            return None
        if time.time() > session.expires_at:
            logger.info("Session %s expired", session.session_id)
            self._close(session, SessionStatus.EXPIRED)
            return None
        return session

    def has_active(self, user_id: str) -> bool:
        return self.get_active(user_id) is not None

    @staticmethod
    def is_global_escape(text: str) -> bool:
        return text.strip().lower().rstrip("!.") in GLOBAL_ESCAPES

    # --------------------------------------------------------------- lifecycle
    async def open(self, workflow, intent: str, entities: dict, user_id: str) -> TurnResult:
        timeout = getattr(workflow, "session_timeout_s", self.default_timeout_s)
        session = Session.new(user_id, workflow.name, timeout)
        self.store.save(session)
        result = await workflow.start(intent, entities or {}, session)
        self._apply(session, result)
        return result

    async def handle(self, user_id: str, text: str) -> TurnResult:
        session = self.get_active(user_id)
        if session is None:
            return TurnResult.complete("")
        workflow = self.workflows.workflows.get(session.workflow_name)
        if workflow is None:
            self._close(session, SessionStatus.CANCELLED)
            return TurnResult.complete("That task is no longer available, sir.")
        result = await workflow.resume(text, session)
        self._apply(session, result)
        return result

    def cancel(self, user_id: str, reason: str = "") -> None:
        session = self.get_active(user_id)
        if session is not None:
            logger.info("Cancelling session %s (%s)", session.session_id, reason)
            self._close(session, SessionStatus.CANCELLED)

    # --------------------------------------------------- background (MT3 hooks)
    def sweep_expired(self):
        """Expire abandoned ACTIVE/AWAITING dialogue sessions. WAITING sessions are
        long-lived and expire only when their owning workflow's on_tick() ends them."""
        now = time.time()
        expired = []
        for session in self.store.list_active_dialogue():
            if now > session.expires_at:
                logger.info("Sweeping expired session %s", session.session_id)
                self._close(session, SessionStatus.EXPIRED)
                expired.append(session)
        return expired

    async def tick_waiting(self):
        """Advance every WAITING session whose wake_at has passed via on_tick().
        Returns the (session, TurnResult) pairs that produced a result to deliver."""
        delivered = []
        now = time.time()
        for session in self.store.list_waiting():
            if session.wake_at is not None and now < session.wake_at:
                continue
            workflow = self.workflows.workflows.get(session.workflow_name)
            if workflow is None:
                self._close(session, SessionStatus.CANCELLED)
                continue
            try:
                result = await workflow.on_tick(session)
            except Exception:
                logger.exception("on_tick failed for session %s", session.session_id)
                continue
            if result is not None:
                self._apply(session, result)
                delivered.append((session, result))
        return delivered

    # ----------------------------------------------------------------- internal
    def _apply(self, session: Session, result: TurnResult) -> None:
        if result.slots_update:
            session.slots.update(result.slots_update)
        if result.next_state:
            session.fsm_state = result.next_state
        session.updated_at = time.time()

        ctrl = result.control
        if ctrl == TurnControl.CONTINUE:
            session.status = SessionStatus.ACTIVE
            session.expires_at = time.time() + session.timeout_s
            self.store.save(session)
        elif ctrl == TurnControl.AWAIT_CONFIRMATION:
            session.status = SessionStatus.AWAITING_CONFIRMATION
            session.expires_at = time.time() + session.timeout_s
            self.store.save(session)
        elif ctrl == TurnControl.BACKGROUND:
            session.status = SessionStatus.WAITING
            session.wake_at = result.wake_at
            self.store.save(session)
        elif ctrl == TurnControl.COMPLETE:
            self._close(session, SessionStatus.DONE)
        elif ctrl == TurnControl.CANCEL:
            self._close(session, SessionStatus.CANCELLED)

        # Keep short-term routing context aligned to the live task.
        if self.context is not None and session.is_active_dialogue:
            self.context.update(session.workflow_name, session.slots)

    def _close(self, session: Session, status: SessionStatus) -> None:
        session.status = status
        session.updated_at = time.time()
        self.store.save(session)
