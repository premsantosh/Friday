"""
SignalNotifier (M4).

Sends a written record of a reservation outcome over Signal via a
signal-cli-rest-api endpoint. Used for async outcomes (and confirmations) so the
user has a record even when not at the mic. Never raises; degrades to a no-op
when unconfigured. The HTTP poster is injectable for testing.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class SignalNotifier:
    def __init__(self, base_url: str, from_number: str, to_number: str,
                 poster: Optional[Callable] = None):
        self.base_url = base_url.rstrip("/")
        self.from_number = from_number
        self.to_number = to_number
        self._poster = poster  # callable(url, json=...) -> response; defaults to requests.post

    @classmethod
    def from_env(cls) -> Optional["SignalNotifier"]:
        base = os.getenv("SIGNAL_CLI_URL")
        frm = os.getenv("SIGNAL_FROM_NUMBER")
        to = os.getenv("SIGNAL_TO_NUMBER")
        if not (base and frm and to):
            return None
        return cls(base, frm, to)

    def send(self, message: str) -> bool:
        """Send `message` to the configured recipient. Returns success; never raises."""
        try:
            poster = self._poster
            if poster is None:
                import requests
                poster = requests.post
            resp = poster(
                f"{self.base_url}/v2/send",
                json={
                    "message": message,
                    "number": self.from_number,
                    "recipients": [self.to_number],
                },
                timeout=10,
            )
            # signal-cli-rest-api returns 2xx on success.
            ok = getattr(resp, "status_code", 200)
            return 200 <= int(ok) < 300
        except Exception:
            logger.warning("Signal notification failed.", exc_info=True)
            return False
