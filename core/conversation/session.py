"""
Layer B core data model — Sessions and turn results.

A `Session` is a durable, possibly long-running multi-turn task owned by a
`ConversationalWorkflow`. The framework never interprets `fsm_state`, `slots`,
or `scratch` — those belong to the workflow. A workflow drives the dialogue by
returning a `TurnResult` whose `control` tells the SessionManager what to do
with the session next.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SessionStatus(Enum):
    ACTIVE = "active"                          # in live dialogue with the user
    AWAITING_CONFIRMATION = "awaiting_confirmation"  # waiting for an explicit yes/no
    WAITING = "waiting"                        # detached; advanced by a background tick
    DONE = "done"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


# Statuses that count as "the active dialogue" (at most one per user).
ACTIVE_STATUSES = (SessionStatus.ACTIVE, SessionStatus.AWAITING_CONFIRMATION)
# Statuses that mean the session is finished and should never take a turn.
TERMINAL_STATUSES = (SessionStatus.DONE, SessionStatus.CANCELLED, SessionStatus.EXPIRED)


class TurnControl(Enum):
    CONTINUE = "continue"                  # stay ACTIVE; message asks for the next slot
    AWAIT_CONFIRMATION = "await_confirmation"  # -> AWAITING_CONFIRMATION; message is the summary
    BACKGROUND = "background"              # -> WAITING; a background tick will advance it
    COMPLETE = "complete"                  # -> DONE; message is the final result
    CANCEL = "cancel"                      # -> CANCELLED


@dataclass
class TurnResult:
    """What a ConversationalWorkflow returns from start()/resume()/on_tick()."""
    message: str
    control: TurnControl = TurnControl.CONTINUE
    slots_update: Optional[Dict[str, Any]] = None   # merged into session.slots
    next_state: Optional[str] = None                # new fsm_state
    wake_at: Optional[float] = None                 # for BACKGROUND: when to next tick
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # --- ergonomic constructors -------------------------------------------------
    @classmethod
    def ask(cls, message: str, *, slots_update: Optional[Dict[str, Any]] = None,
            next_state: Optional[str] = None, **kw) -> "TurnResult":
        return cls(message=message, control=TurnControl.CONTINUE,
                   slots_update=slots_update, next_state=next_state, **kw)

    @classmethod
    def confirm(cls, message: str, *, slots_update: Optional[Dict[str, Any]] = None,
                next_state: Optional[str] = None, **kw) -> "TurnResult":
        return cls(message=message, control=TurnControl.AWAIT_CONFIRMATION,
                   slots_update=slots_update, next_state=next_state, **kw)

    @classmethod
    def background(cls, message: str, *, wake_at: Optional[float] = None,
                   slots_update: Optional[Dict[str, Any]] = None,
                   next_state: Optional[str] = None, **kw) -> "TurnResult":
        return cls(message=message, control=TurnControl.BACKGROUND, wake_at=wake_at,
                   slots_update=slots_update, next_state=next_state, **kw)

    @classmethod
    def complete(cls, message: str, *, slots_update: Optional[Dict[str, Any]] = None,
                 **kw) -> "TurnResult":
        return cls(message=message, control=TurnControl.COMPLETE,
                   slots_update=slots_update, **kw)

    @classmethod
    def cancel(cls, message: str, **kw) -> "TurnResult":
        return cls(message=message, control=TurnControl.CANCEL, **kw)


@dataclass
class Session:
    session_id: str
    user_id: str
    workflow_name: str
    fsm_state: str = "start"
    slots: Dict[str, Any] = field(default_factory=dict)
    scratch: Dict[str, Any] = field(default_factory=dict)
    status: SessionStatus = SessionStatus.ACTIVE
    timeout_s: int = 600
    wake_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: float = 0.0

    @staticmethod
    def new(user_id: str, workflow_name: str, timeout_s: int) -> "Session":
        now = time.time()
        return Session(
            session_id=uuid.uuid4().hex,
            user_id=user_id,
            workflow_name=workflow_name,
            timeout_s=timeout_s,
            created_at=now,
            updated_at=now,
            expires_at=now + timeout_s,
        )

    @property
    def is_active_dialogue(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES
