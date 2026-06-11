"""Yelp channel — Playwright + the user's logged-in session.

Yelp Fusion is used for discovery (see discovery.py); the booking write goes
through the site with the user's session. Site-specific flow is layered in later.
"""

from __future__ import annotations

from ..models import ReservationMethod
from .base import BrowserChannel


class YelpChannel(BrowserChannel):
    def __init__(self, profile_dir: str):
        super().__init__(ReservationMethod.YELP, "yelp", profile_dir)
