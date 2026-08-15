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
    # Google Calendar service-account inserter
    "GOOGLE_APPLICATION_CREDENTIALS",
    # Bland.ai autonomous phone calls
    "BLAND_API_KEY",
    # Outbound email (SMTP) + reply polling (IMAP)
    "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "RESERVATION_EMAIL_FROM",
    "IMAP_HOST", "IMAP_USER", "IMAP_PASSWORD",
    # Research substrate (research/): writes ~/.friday/research.db and calls
    # local Ollama when enabled — must never turn on implicitly in tests.
    "FRIDAY_RESEARCH",
)


@pytest.fixture(autouse=True)
def isolate_external_services(monkeypatch):
    """Strip live external-service credentials so tests never hit them by default."""
    for var in _EXTERNAL_SERVICE_ENV:
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture(autouse=True)
def no_results_writes_into_the_repo(tmp_path, monkeypatch):
    """Point research/report.py's default output at tmp_path.

    The eval CSV is the study's longitudinal record. A test that forgets to pass
    results_dir would otherwise append synthetic FakeJudge rows to the real
    results/eval.csv, which is a quietly corrupted dataset rather than a visible
    failure. Tests that assert on output pass results_dir explicitly.
    """
    from research import report

    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path / "repo-results")
    yield
