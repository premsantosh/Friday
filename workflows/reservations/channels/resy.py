"""Resy channel — Playwright + the user's logged-in session.

Shares the BrowserChannel lifecycle/confirmation contract; the site-specific
booking flow is layered in later (requires a logged-in Resy account).
"""

from __future__ import annotations

from ..models import ReservationMethod
from .base import BrowserChannel


class ResyChannel(BrowserChannel):
    def __init__(self, profile_dir: str):
        super().__init__(ReservationMethod.RESY, "resy", profile_dir)
