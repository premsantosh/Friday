"""Shared pytest fixtures for the test suite.

Keeps tests hermetic: the developer's real shell environment configures live
external services (Google Calendar, Telegram, Bland.ai phone calls), and several
reservation tests drive a *successful* booking — which fires
ReservationWorkflow._on_confirmed → real calendar event + real Telegram message.
Without isolation those tests spam the developer's actual calendar and phone.

This autouse fixture removes the side-effecting credentials before every test so
`CalendarService.from_env()` / `TelegramNotifier.from_env()` / the Bland phone
client degrade to safe no-ops. Tests that need a value set it themselves via
`monkeypatch.setenv`, which runs after this fixture and overrides it; tests that
want to verify calendar/Telegram *are* called inject their own recording fakes
(see test_confirmed_booking_creates_calendar_and_telegram).
"""

import pytest

# Credentials whose mere presence makes a `from_env()` service reach the network
# or place real, user-visible side effects.
_EXTERNAL_SERVICE_ENV = (
    # Telegram bot channel + reservation notifier
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_CHAT_IDS", "TELEGRAM_NOTIFY_CHAT_ID",
    # Voice PE satellite devices (ESPHome native API connections)
    "VOICE_PE_DEVICES", "VOICE_PE_NOISE_PSK",
    # Google Calendar service-account inserter
    "GOOGLE_APPLICATION_CREDENTIALS",
    # Bland.ai autonomous phone calls
    "BLAND_API_KEY",
    # Outbound email (SMTP) + reply polling (IMAP)
    "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "RESERVATION_EMAIL_FROM",
    "IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD",
    # TableCheck watcher: a developer's endpoint override would change the URLs
    # the fixture-driven channel tests assert on.
    "TABLECHECK_AVAILABILITY_URL",
)


@pytest.fixture(autouse=True)
def isolate_external_services(monkeypatch):
    """Strip live external-service credentials so tests never hit them by default."""
    for var in _EXTERNAL_SERVICE_ENV:
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture(autouse=True)
def isolate_profile_store(monkeypatch, tmp_path):
    """Point the durable user profile at a throwaway database.

    `UserProfile()` otherwise opens the developer's real ~/.friday/memory.db,
    and the form tests write to it — a test run would quietly edit the facts
    Friday uses to fill live booking forms.
    """
    monkeypatch.setenv("FRIDAY_PROFILE_DB", str(tmp_path / "profile.db"))
    # The form agent dumps a screenshot when a submission can't be confirmed;
    # keep those out of the developer's real ~/.friday/browser-debug.
    monkeypatch.setenv("RESERVATION_BROWSER_DEBUG_DIR", str(tmp_path / "browser-debug"))
    yield
