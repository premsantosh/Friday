"""OpenTable channel — Playwright + the user's logged-in session.

Inherits the BrowserChannel lifecycle/confirmation contract. The site-specific
booking flow (slot grid + logged-in submit) is the remaining live-tuning work;
until implemented it hands off safely via the base default `_do_booking`.
"""

from __future__ import annotations

from ..models import ReservationMethod
from .base import BrowserChannel


class OpenTableChannel(BrowserChannel):
    def __init__(self, profile_dir: str):
        super().__init__(ReservationMethod.OPENTABLE, "opentable", profile_dir)
