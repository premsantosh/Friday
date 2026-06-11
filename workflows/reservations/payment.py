"""
PrivacyCardService (M3).

Mints **single-use** virtual cards via the Privacy.com API
(https://developers.privacy.com/docs/getting-started), **hard-capped at $10** —
the code refuses to mint above the cap, full stop. The returned card is passed
transiently to a channel's commit() for one transaction and is **never logged,
never sent to the LLM, and never persisted** (only Privacy's opaque token is
safe to keep). See docs/reservations-agent-spec.md §8.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

HARD_CAP_USD = 10.0


@dataclass
class VirtualCard:
    token: str          # Privacy's opaque reference (safe to keep)
    pan: str            # sensitive — never log / persist / send to an LLM
    cvv: str            # sensitive
    exp_month: str
    exp_year: str
    last_four: str
    spend_limit_usd: float

    def __repr__(self) -> str:  # guard against accidental leakage in logs/tracebacks
        return f"VirtualCard(last_four={self.last_four!r}, limit=${self.spend_limit_usd:.2f})"

    __str__ = __repr__


class PrivacyCardService:
    def __init__(self, api_key: str, limit_usd: float = HARD_CAP_USD,
                 base_url: str = "https://api.privacy.com/v1"):
        self.api_key = api_key
        # The configured limit can only ever lower the ceiling, never raise it.
        self.limit_usd = min(float(limit_usd), HARD_CAP_USD)
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> Optional["PrivacyCardService"]:
        key = os.getenv("PRIVACY_API_KEY")
        if not key:
            return None
        try:
            limit = float(os.getenv("RESERVATION_CARD_LIMIT_USD", str(HARD_CAP_USD)))
        except ValueError:
            limit = HARD_CAP_USD
        base = os.getenv("PRIVACY_API_BASE_URL", "https://api.privacy.com/v1")
        return cls(key, limit_usd=limit, base_url=base)

    def mint_single_use(self, amount_usd: Optional[float] = None,
                        memo: str = "Friday reservation") -> Optional[VirtualCard]:
        """Create a single-use card. Returns None on refusal (over cap) or any API failure."""
        amount = self.limit_usd if amount_usd is None else float(amount_usd)

        if amount > HARD_CAP_USD:
            # Hard cap: never mint above $10, even if asked to.
            logger.error("Refusing to mint a card above the $%.2f cap (requested $%.2f).",
                         HARD_CAP_USD, amount)
            return None
        amount = min(amount, self.limit_usd)
        cents = int(round(amount * 100))

        try:
            import requests

            resp = requests.post(
                f"{self.base_url}/card",
                headers={
                    "Authorization": f"api-key {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "type": "SINGLE_USE",
                    "spend_limit": cents,
                    "spend_limit_duration": "TRANSACTION",
                    "memo": memo[:50],
                    "state": "OPEN",
                },
                timeout=10,
            )
            resp.raise_for_status()
            d = resp.json()
            return VirtualCard(
                token=d.get("token", ""),
                pan=d.get("pan", ""),
                cvv=d.get("cvv", ""),
                exp_month=str(d.get("exp_month", "")),
                exp_year=str(d.get("exp_year", "")),
                last_four=d.get("last_four", ""),
                spend_limit_usd=amount,
            )
        except Exception:
            # Never include request/response bodies here — they carry card data.
            logger.warning("Privacy card mint failed.", exc_info=True)
            return None
