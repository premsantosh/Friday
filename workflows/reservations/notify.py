"""
TelegramNotifier (M4).

Sends a written record of a reservation outcome over Telegram via the Bot API.
Used for async outcomes (and confirmations) so the user has a record even when
not at the mic. Never raises; degrades to a no-op when unconfigured. The HTTP
poster is injectable for testing.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, token: str, chat_id, poster: Optional[Callable] = None):
        self.api = f"https://api.telegram.org/bot{token}"
        self.chat_id = chat_id
        self._poster = poster  # callable(url, json=...) -> response; defaults to requests.post

    @classmethod
    def from_env(cls) -> Optional["TelegramNotifier"]:
        """Build from env, or None when unconfigured.

        Reuses TELEGRAM_BOT_TOKEN (from @BotFather). The recipient is
        TELEGRAM_NOTIFY_CHAT_ID, falling back to the first id in
        TELEGRAM_ALLOWED_CHAT_IDS so a single-user setup needs no extra config.
        """
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_NOTIFY_CHAT_ID")
        if not chat_id:
            allowed = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
            chat_id = next((c.strip() for c in allowed.split(",") if c.strip()), None)
        if not (token and chat_id):
            return None
        return cls(token, chat_id)

    def send(self, message: str) -> bool:
        """Send `message` to the configured chat. Returns success; never raises."""
        try:
            poster = self._poster
            if poster is None:
                import requests
                poster = requests.post
            resp = poster(
                f"{self.api}/sendMessage",
                json={"chat_id": self.chat_id, "text": message},
                timeout=10,
            )
            ok = getattr(resp, "status_code", 200)
            return 200 <= int(ok) < 300
        except Exception:
            logger.warning("Telegram notification failed.", exc_info=True)
            return False
