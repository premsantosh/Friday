"""
Tests for the two-way Telegram channel (core/telegram_channel.py).

Hermetic: HTTP is injected via the `poster`/`getter` hooks, so nothing touches
the Telegram Bot API. The autouse fixture in conftest.py strips TELEGRAM_* env
so from_env() tests start clean.
"""

from __future__ import annotations

import asyncio

from core.telegram_channel import TelegramChannel


class FakeResp:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload if payload is not None else {"ok": True, "result": []}
        self.status_code = status_code

    def json(self):
        return self._payload


def _update(update_id, chat_id, text, kind="message"):
    return {"update_id": update_id, kind: {"chat": {"id": chat_id}, "text": text}}


def _ok(updates):
    return {"ok": True, "result": updates}


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# --------------------------------------------------------------------- parsing

def test_extract_messages_text_and_edited():
    result = [
        _update(1, 555, "hello"),
        _update(2, 777, "edited!", kind="edited_message"),
    ]
    assert TelegramChannel._extract_messages(result) == [("555", "hello"), ("777", "edited!")]


def test_extract_messages_skips_non_text():
    result = [
        {"update_id": 1, "message": {"chat": {"id": 555}}},                 # no text
        {"update_id": 2, "message": {"chat": {"id": 555}, "text": "  "}},   # blank
        {"update_id": 3, "callback_query": {"data": "x"}},                  # not a message
        _update(4, 555, "real"),
    ]
    assert TelegramChannel._extract_messages(result) == [("555", "real")]


# --------------------------------------------------------------------- outbound

def test_send_posts_expected_payload():
    sent = {}

    def poster(url, json=None, timeout=None):
        sent["url"] = url
        sent["json"] = json
        return FakeResp(status_code=200)

    ch = TelegramChannel("TOK", poster=poster)
    assert ch.send("done, sir", "555") is True
    assert sent["url"] == "https://api.telegram.org/botTOK/sendMessage"
    assert sent["json"] == {"chat_id": "555", "text": "done, sir"}


def test_send_reports_failure_without_raising():
    ch = TelegramChannel("TOK", poster=lambda url, json=None, timeout=None: FakeResp(status_code=403))
    assert ch.send("x", "555") is False


# --------------------------------------------------------------------- dispatch

def _channel_with_updates(updates, allowed, replies, *, skip_backlog=False):
    """Build a channel whose getter returns `updates` once then empties."""
    calls = {"n": 0}

    def getter(url, params=None, timeout=None):
        calls["n"] += 1
        return FakeResp(_ok(updates if calls["n"] == 1 else []))

    def poster(url, json=None, timeout=None):
        replies.append(json)
        return FakeResp(status_code=200)

    ch = TelegramChannel("TOK", allowed_chat_ids=allowed, getter=getter, poster=poster)
    ch._skip_backlog = skip_backlog
    return ch, calls


def test_poll_routes_authorised_chat_and_replies():
    replies = []
    ch, _ = _channel_with_updates([_update(10, 555, "what time is it?")], ["555"], replies)

    seen = []

    async def handler(text, chat_id):
        seen.append((text, chat_id))
        return "It is noon, sir."

    ch._handler = handler
    _run(ch._poll_once())

    assert seen == [("what time is it?", "555")]
    assert replies == [{"chat_id": "555", "text": "It is noon, sir."}]
    assert ch._offset == 11  # advanced past update_id 10


def test_poll_ignores_unauthorised_chat():
    replies = []
    ch, _ = _channel_with_updates([_update(10, 999, "delete everything")], ["555"], replies)

    called = []

    async def handler(text, chat_id):
        called.append(chat_id)
        return "nope"

    ch._handler = handler
    _run(ch._poll_once())
    assert called == []
    assert replies == []          # no command reply
    assert ch._offset == 11       # still acknowledged so it won't repeat


def test_poll_bootstrap_reply_when_no_allowlist():
    replies = []
    ch, _ = _channel_with_updates([_update(10, 555, "hi")], [], replies)

    called = []

    async def handler(text, chat_id):
        called.append(chat_id)
        return "should not run"

    ch._handler = handler
    _run(ch._poll_once())

    assert called == []                       # command not executed
    assert len(replies) == 1
    assert replies[0]["chat_id"] == "555"
    assert "555" in replies[0]["text"] and "TELEGRAM_ALLOWED_CHAT_IDS" in replies[0]["text"]


def test_poll_skips_backlog_on_first_poll():
    replies = []
    ch, _ = _channel_with_updates(
        [_update(10, 555, "stale command")], ["555"], replies, skip_backlog=True
    )

    called = []

    async def handler(text, chat_id):
        called.append(chat_id)
        return "x"

    ch._handler = handler
    _run(ch._poll_once())

    assert called == []          # backlog not dispatched
    assert replies == []
    assert ch._offset == 11       # but acknowledged so the next poll is fresh
    assert ch._skip_backlog is False


def test_backlog_drain_is_non_blocking_then_long_polls():
    """First poll must use timeout=0 so a live message sent right after startup
    isn't scooped into the discarded backlog batch; later polls long-poll."""
    seen_timeouts = []

    def getter(url, params=None, timeout=None):
        seen_timeouts.append(params.get("timeout"))
        return FakeResp(_ok([]))

    ch = TelegramChannel("TOK", allowed_chat_ids=["555"], long_poll_seconds=25, getter=getter)
    ch._handler = lambda text, chat_id: None
    _run(ch._poll_once())  # startup drain
    _run(ch._poll_once())  # steady state
    assert seen_timeouts == [0, 25]


def test_poll_409_conflict_is_handled_quietly():
    """A second poller gets HTTP 409 — must not raise or dispatch."""
    def getter(url, params=None, timeout=None):
        return FakeResp(_ok([]), status_code=409)

    ch = TelegramChannel("TOK", allowed_chat_ids=["555"], getter=getter)
    ch._skip_backlog = False
    ch._handler = lambda text, chat_id: None
    _run(ch._poll_once())  # must not raise
    assert ch._offset is None


def test_poll_handles_not_ok_payload():
    def getter(url, params=None, timeout=None):
        return FakeResp({"ok": False, "error_code": 401, "description": "Unauthorized"})

    ch = TelegramChannel("BADTOK", allowed_chat_ids=["555"], getter=getter)
    ch._skip_backlog = False
    ch._handler = lambda text, chat_id: None
    _run(ch._poll_once())  # must not raise
    assert ch._offset is None


def test_handler_exception_sends_error_and_does_not_crash():
    replies = []
    ch, _ = _channel_with_updates([_update(10, 555, "boom")], ["555"], replies)

    async def handler(text, chat_id):
        raise RuntimeError("kaboom")

    ch._handler = handler
    _run(ch._poll_once())
    assert len(replies) == 1 and "error" in replies[0]["text"].lower()


# --------------------------------------------------------------------- from_env

def test_from_env_returns_none_when_unconfigured():
    assert TelegramChannel.from_env() is None


def test_from_env_with_token_and_allowlist(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "555, 777")
    ch = TelegramChannel.from_env()
    assert ch is not None
    assert ch.allowed_chat_ids == {"555", "777"}
    assert ch.api == "https://api.telegram.org/bot123:ABC"


def test_from_env_token_only_empty_allowlist(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    ch = TelegramChannel.from_env()
    assert ch is not None
    assert ch.allowed_chat_ids == set()
