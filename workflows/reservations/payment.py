"""
Payment services (M3).

Two providers behind one interface (`mint_single_use() -> Optional[VirtualCard]`):

  - **PrivacyCardService** — mints single-use virtual cards via the Privacy.com
    API (https://developers.privacy.com/docs/getting-started).
  - **ManualCardService** — a card the user minted **themselves** at Privacy.com
    and pasted into `.env` (no API subscription needed). Same trust rules apply:
    use a SINGLE-USE card capped at the cap or less — never a real card — and
    replace it once spent.

Both are **hard-capped at $10** — over-cap requests are refused in code, and
the gate's SpendCapPolicy enforces it again independently. The returned card is
passed transiently to a channel's commit() for one transaction and is **never
logged, never sent to the LLM, and never persisted** (only an opaque token is
safe to keep). See docs/reservations-agent-spec.md §8.

`card_service_from_env()` picks the provider: RESERVATION_CARD_PROVIDER
(privacy | manual | off), defaulting to whichever is configured.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from core.harness.egress import luhn_ok

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
    zip_code: str = ""  # billing ZIP some checkouts require (Privacy ignores AVS)

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


class ManualCardService:
    """A user-provided card from the environment (RESERVATION_CARD_NUMBER /
    _CVV / _EXP_MONTH / _EXP_YEAR), for when the Privacy.com API isn't
    subscribed but the user has minted a single-use card by hand.

    The same card is returned for every "mint" while the process runs — it's
    the user's job to replace it after it's spent (a true single-use Privacy
    card declines on reuse anyway, so the failure mode is a safe decline).
    Card values never leave .env except transiently into a channel's commit.
    """

    def __init__(self, pan: str, cvv: str, exp_month: str, exp_year: str,
                 limit_usd: float = HARD_CAP_USD, zip_code: str = ""):
        self.pan = pan
        self.cvv = cvv
        self.exp_month = exp_month
        self.exp_year = exp_year
        self.zip_code = zip_code
        # The configured limit can only ever lower the ceiling, never raise it.
        self.limit_usd = min(float(limit_usd), HARD_CAP_USD)

    def __repr__(self) -> str:  # never expose the PAN in logs/tracebacks
        return f"ManualCardService(last_four={self.pan[-4:]!r})"

    __str__ = __repr__

    @classmethod
    def from_env(cls) -> Optional["ManualCardService"]:
        pan = re.sub(r"[ \-]", "", os.getenv("RESERVATION_CARD_NUMBER", ""))
        cvv = os.getenv("RESERVATION_CARD_CVV", "").strip()
        month = os.getenv("RESERVATION_CARD_EXP_MONTH", "").strip()
        year = os.getenv("RESERVATION_CARD_EXP_YEAR", "").strip()
        if not pan:
            return None
        if not (pan.isdigit() and 13 <= len(pan) <= 19 and luhn_ok(pan)):
            logger.error("RESERVATION_CARD_NUMBER is not a valid card number; "
                         "manual card disabled.")
            return None
        if not re.fullmatch(r"\d{3,4}", cvv):
            logger.error("RESERVATION_CARD_CVV missing/invalid; manual card disabled.")
            return None
        if not (month.isdigit() and 1 <= int(month) <= 12 and
                re.fullmatch(r"\d{2}|\d{4}", year)):
            logger.error("RESERVATION_CARD_EXP_MONTH/_EXP_YEAR missing/invalid; "
                         "manual card disabled.")
            return None
        try:
            limit = float(os.getenv("RESERVATION_CARD_LIMIT_USD", str(HARD_CAP_USD)))
        except ValueError:
            limit = HARD_CAP_USD
        if len(year) == 2:
            year = "20" + year
        zip_code = os.getenv("RESERVATION_CARD_ZIP", "").strip()
        return cls(pan, cvv, month.zfill(2), year, limit_usd=limit, zip_code=zip_code)

    def mint_single_use(self, amount_usd: Optional[float] = None,
                        memo: str = "Friday reservation") -> Optional[VirtualCard]:
        """Hand out the env-provided card. Same cap semantics as the API path."""
        amount = self.limit_usd if amount_usd is None else float(amount_usd)
        if amount > HARD_CAP_USD:
            logger.error("Refusing to use the manual card above the $%.2f cap "
                         "(requested $%.2f).", HARD_CAP_USD, amount)
            return None
        logger.info("Using the manually configured single-use card (•%s). "
                    "Remember to replace it once spent.", self.pan[-4:])
        return VirtualCard(
            token="manual",
            pan=self.pan,
            cvv=self.cvv,
            exp_month=self.exp_month,
            exp_year=self.exp_year,
            last_four=self.pan[-4:],
            spend_limit_usd=min(amount, self.limit_usd),
            zip_code=self.zip_code,
        )


def card_service_from_env():
    """Provider selection: RESERVATION_CARD_PROVIDER = privacy | manual | off.
    Unset → privacy when PRIVACY_API_KEY exists, else manual when a card is
    configured, else no payment service."""
    provider = os.getenv("RESERVATION_CARD_PROVIDER", "").strip().lower()
    if provider == "off":
        return None
    if provider == "privacy":
        return PrivacyCardService.from_env()
    if provider == "manual":
        return ManualCardService.from_env()
    return PrivacyCardService.from_env() or ManualCardService.from_env()
