"""ConversationRecorder: the single place exchanges and feedback enter research.db.

Two entry points feed it, both fire-and-forget from the caller's perspective:

  * The LLM provider calls `record_chat` for real free-chat exchanges (it alone
    has the augmented system prompt for the context snapshot) and this triggers
    the live shadow generation.
  * The assistant calls `record_turn` for every processed input with the route
    taken and total latency. When the provider already recorded the same
    exchange moments earlier, `record_turn` backfills metadata instead of
    inserting a duplicate.

It also owns Telegram feedback: building the 👍/👎 inline keyboard for fresh
chat replies and handling the button callbacks.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional, TYPE_CHECKING

from research.db import ResearchStore

if TYPE_CHECKING:
    from research.shadow import ShadowRunner

logger = logging.getLogger(__name__)

# A provider-recorded exchange and the assistant-level record of the same turn
# arrive within one process_input call; anything older is a different turn.
_DEDUPE_WINDOW_S = 5.0
# Only offer feedback buttons for the reply just sent, not stale exchanges.
_FRESH_WINDOW_S = 30.0


class ConversationRecorder:
    def __init__(
        self,
        store: ResearchStore,
        shadow: Optional["ShadowRunner"] = None,
        feedback_buttons: bool = True,
    ):
        self.store = store
        self.shadow = shadow
        self.feedback_buttons = feedback_buttons
        self._lock = threading.Lock()
        # (exchange_id, user_text, reply_text, monotonic) of the last provider record
        self._last_chat: Optional[tuple[int, str, str, float]] = None
        # user_id -> (exchange_id, monotonic) of their last *chat* exchange
        self._last_for_user: dict[str, tuple[int, float]] = {}

    # ------------------------------------------------------------- recording
    def record_chat(
        self,
        user_text: str,
        reply_text: str,
        *,
        model: Optional[str] = None,
        context_snapshot: Optional[dict] = None,
        memory_turn_id: Optional[int] = None,
    ) -> int:
        """Called by the LLM provider on a real chat exchange."""
        exchange_id = self.store.record_exchange(
            user_text,
            reply_text,
            route="chat",
            model=model,
            context_snapshot=context_snapshot,
            memory_turn_id=memory_turn_id,
        )
        with self._lock:
            self._last_chat = (exchange_id, user_text, reply_text, time.monotonic())
        if self.shadow is not None:
            self.shadow.enqueue(exchange_id)
        return exchange_id

    def record_turn(
        self,
        user_text: str,
        reply_text: str,
        *,
        route: str,
        latency_ms: Optional[int] = None,
        user_id: str = "default",
        channel: Optional[str] = None,
    ) -> int:
        """Called by the assistant after every processed input."""
        now = time.monotonic()
        with self._lock:
            last = self._last_chat
        if (
            last is not None
            and last[1] == user_text
            and last[2] == reply_text
            and now - last[3] < _DEDUPE_WINDOW_S
        ):
            exchange_id = last[0]
            self.store.update_exchange(
                exchange_id, channel=channel, user_id=user_id, latency_ms=latency_ms
            )
            is_chat = True
        else:
            exchange_id = self.store.record_exchange(
                user_text,
                reply_text,
                route=route,
                channel=channel,
                user_id=user_id,
                latency_ms=latency_ms,
            )
            is_chat = route == "chat"
        if is_chat:
            with self._lock:
                self._last_for_user[user_id] = (exchange_id, now)
        return exchange_id

    # -------------------------------------------------------------- feedback
    def feedback_markup(self, chat_id: str, reply_text: str) -> Optional[dict]:
        """Inline 👍/👎 keyboard for the reply just produced, or None.

        Only chat exchanges get buttons (workflow confirmations don't), and only
        while fresh — a slow workflow reply arriving later must not pick up the
        keyboard of an older chat exchange.
        """
        if not self.feedback_buttons:
            return None
        with self._lock:
            entry = self._last_for_user.get(str(chat_id))
        if entry is None or time.monotonic() - entry[1] > _FRESH_WINDOW_S:
            return None
        exchange_id = entry[0]
        return {
            "inline_keyboard": [[
                {"text": "👍", "callback_data": f"fb:{exchange_id}:1"},
                {"text": "👎", "callback_data": f"fb:{exchange_id}:0"},
            ]]
        }

    def handle_callback(self, chat_id: str, data: str) -> None:
        """Handle a Telegram callback_query payload like 'fb:<exchange_id>:<1|0>'."""
        try:
            tag, raw_id, raw_sig = data.split(":", 2)
            if tag != "fb":
                return
            exchange_id = int(raw_id)
            signal = 1 if raw_sig == "1" else -1
        except (ValueError, AttributeError):
            logger.debug("Ignoring malformed callback data: %r", data)
            return
        self.store.add_feedback(
            exchange_id,
            kind="explicit",
            signal=signal,
            source="telegram_button",
            details=json.dumps({"chat_id": str(chat_id)}),
        )
