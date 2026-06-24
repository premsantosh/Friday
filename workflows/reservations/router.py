"""
ChannelRouter — maps a ChannelDecision to the channel that will book it.

M2 wires the browser channels (OpenTable/Resy/Yelp/GenericWeb). Phone (M5),
email (M5b), and the sandbox-bot fallback (M7) are not selectable yet — those
methods return None and the workflow simply describes the plan for now.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from .channels import (
    BlandClient,
    DialAndBridgeChannel,
    EmailChannel,
    GenericWebChannel,
    OpenTableChannel,
    PhoneChannel,
    ReservationChannel,
    ResyChannel,
    YelpChannel,
)
from .llm import ReservationLLM, make_llm_drafter
from .models import ChannelDecision, ReservationMethod


class ChannelRouter:
    def __init__(self, channels: Optional[Dict[ReservationMethod, ReservationChannel]] = None):
        self._channels = channels or {}

    @classmethod
    def from_env(cls) -> "ChannelRouter":
        browser_dir = os.path.expanduser(
            os.getenv("RESERVATION_BROWSER_DIR", "~/.friday/browser")
        )
        channels: Dict[ReservationMethod, ReservationChannel] = {
            ReservationMethod.OPENTABLE: OpenTableChannel(browser_dir),
            ReservationMethod.RESY: ResyChannel(browser_dir),
            ReservationMethod.YELP: YelpChannel(browser_dir),
            ReservationMethod.GENERIC_WEB: GenericWebChannel(browser_dir),
        }

        phone = cls._phone_channel_from_env()
        if phone is not None:
            channels[ReservationMethod.PHONE] = phone

        email = EmailChannel.from_env()
        if email is not None:
            llm = ReservationLLM.from_env()
            if llm is not None:
                email.drafter = make_llm_drafter(llm, fallback=email.drafter)
            channels[ReservationMethod.EMAIL] = email

        return cls(channels)

    @staticmethod
    def _phone_channel_from_env() -> Optional[ReservationChannel]:
        provider = os.getenv("RESERVATION_PHONE_PROVIDER", "bland").lower()
        if provider == "off":
            return None
        callback = os.getenv("RESERVATION_USER_PHONE", "")
        guest = os.getenv("RESERVATION_GUEST_NAME", "")
        if provider == "dial_and_bridge":
            return DialAndBridgeChannel(callback_number=callback, guest_name=guest)
        client = BlandClient.from_env()
        if client is None:
            return None  # bland selected but no API key → not bookable
        return PhoneChannel(client=client, callback_number=callback, guest_name=guest)

    def select(self, decision: ChannelDecision) -> Optional[ReservationChannel]:
        """Return the channel for the decided method, or None if not bookable yet."""
        return self._channels.get(decision.method)
