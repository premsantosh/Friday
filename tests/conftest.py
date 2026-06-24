"""Shared pytest fixtures for the test suite.

Keeps tests hermetic: the developer's real shell environment configures live
external services (Google Calendar, Signal, Bland.ai phone calls), and several
reservation tests drive a *successful* booking — which fires
ReservationWorkflow._on_confirmed → real calendar event + real Signal message.
Without isolation those tests spam the developer's actual calendar and phone.

This autouse fixture removes the side-effecting credentials before every test so
`CalendarService.from_env()` / `SignalNotifier.from_env()` / the Bland phone
client degrade to safe no-ops. Tests that need a value set it themselves via
`monkeypatch.setenv`, which runs after this fixture and overrides it; tests that
want to verify calendar/Signal *are* called inject their own recording fakes
(see test_confirmed_booking_creates_calendar_and_signal).
"""

import pytest

# Credentials whose mere presence makes a `from_env()` service reach the network
# or place real, user-visible side effects.
_EXTERNAL_SERVICE_ENV = (
    # Signal notifications (signal-cli-rest-api)
    "SIGNAL_CLI_URL", "SIGNAL_FROM_NUMBER", "SIGNAL_TO_NUMBER",
    # Google Calendar service-account inserter
    "GOOGLE_APPLICATION_CREDENTIALS",
    # Bland.ai autonomous phone calls
    "BLAND_API_KEY",
    # Outbound email (SMTP) + reply polling (IMAP)
    "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "RESERVATION_EMAIL_FROM",
    "IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD",
)


@pytest.fixture(autouse=True)
def isolate_external_services(monkeypatch):
    """Strip live external-service credentials so tests never hit them by default."""
    for var in _EXTERNAL_SERVICE_ENV:
        monkeypatch.delenv(var, raising=False)
    yield
