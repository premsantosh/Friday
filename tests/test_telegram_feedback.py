"""Tests for Telegram feedback buttons + callback handling (research substrate).

Extends the FakeResp/poster pattern from test_telegram_channel.py. Hermetic:
injected HTTP, no network. Also regression-checks that a default-constructed
channel behaves exactly as before the hooks existed.
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


def _msg_update(update_id, chat_id, text):
    return {"update_id": update_id, "message": {"chat": {"id": chat_id}, "text": text}}


def _cb_update(update_id, chat_id, data, callback_id="cbq1"):
    return {"update_id": update_id, "callback_query": {
        "id": callback_id,
        "data": data,
        "message": {"chat": {"id": chat_id}},
    }}


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _channel(updates, allowed, posts):
    calls = {"n": 0}

    def getter(url, params=None, timeout=None):
        calls["n"] += 1
        return FakeResp({"ok": True, "result": updates if calls["n"] == 1 else []})

    def poster(url, json=None, timeout=None):
        posts.append((url, json))
        return FakeResp(status_code=200)

    ch = TelegramChannel("TOK", allowed_chat_ids=allowed, getter=getter, poster=poster)
    ch._skip_backlog = False
    return ch


# --------------------------------------------------------------------- parsing

def test_extract_callbacks():
    result = [
        _cb_update(1, 555, "fb:42:1"),
        {"update_id": 2, "callback_query": {"id": "x"}},        # no data/chat
        _msg_update(3, 555, "plain message"),
        {"update_id": 4, "not_a_dict_update": True},
    ]
    assert TelegramChannel._extract_callbacks(result) == [("cbq1", "555", "fb:42:1")]


def test_extract_callbacks_non_list():
    assert TelegramChannel._extract_callbacks(None) == []


# ------------------------------------------------------------------- callbacks

def test_callback_answered_and_handler_invoked():
    posts, seen = [], []
    ch = _channel([_cb_update(1, 555, "fb:42:1")], ["555"], posts)
    ch.on_callback = lambda chat_id, data: seen.append((chat_id, data))

    _run(ch._poll_once())

    assert seen == [("555", "fb:42:1")]
    answer_posts = [p for p in posts if p[0].endswith("/answerCallbackQuery")]
    assert answer_posts == [(f"{ch.api}/answerCallbackQuery", {"callback_query_id": "cbq1"})]


def test_callback_from_unauthorised_chat_answered_but_not_handled():
    posts, seen = [], []
    ch = _channel([_cb_update(1, 999, "fb:42:1")], ["555"], posts)
    ch.on_callback = lambda chat_id, data: seen.append((chat_id, data))

    _run(ch._poll_once())

    assert seen == []
    assert any(p[0].endswith("/answerCallbackQuery") for p in posts)


def test_callback_handler_exception_is_contained():
    posts = []
    ch = _channel([_cb_update(1, 555, "fb:42:0")], ["555"], posts)

    def exploding(chat_id, data):
        raise RuntimeError("boom")

    ch.on_callback = exploding
    _run(ch._poll_once())  # must not raise


def test_callbacks_ignored_without_handler():
    posts = []
    ch = _channel([_cb_update(1, 555, "fb:42:1")], ["555"], posts)
    _run(ch._poll_once())  # no on_callback set — answered, nothing else
    assert any(p[0].endswith("/answerCallbackQuery") for p in posts)


# --------------------------------------------------------------- reply markup

def test_reply_gets_markup_from_feedback_provider():
    posts = []
    ch = _channel([_msg_update(1, 555, "hello")], ["555"], posts)
    markup = {"inline_keyboard": [[{"text": "👍", "callback_data": "fb:1:1"}]]}
    ch.feedback_provider = lambda chat_id, reply: markup

    async def handler(text, chat_id):
        return "Good evening, sir."

    ch._handler = handler
    _run(ch._poll_once())

    sends = [p for p in posts if p[0].endswith("/sendMessage")]
    assert sends == [(f"{ch.api}/sendMessage",
                      {"chat_id": "555", "text": "Good evening, sir.", "reply_markup": markup})]


def test_reply_without_provider_has_no_markup():
    posts = []
    ch = _channel([_msg_update(1, 555, "hello")], ["555"], posts)

    async def handler(text, chat_id):
        return "Good evening, sir."

    ch._handler = handler
    _run(ch._poll_once())

    sends = [p for p in posts if p[0].endswith("/sendMessage")]
    assert sends == [(f"{ch.api}/sendMessage", {"chat_id": "555", "text": "Good evening, sir."})]


def test_feedback_provider_exception_still_sends_reply():
    posts = []
    ch = _channel([_msg_update(1, 555, "hello")], ["555"], posts)

    def exploding(chat_id, reply):
        raise RuntimeError("boom")

    ch.feedback_provider = exploding

    async def handler(text, chat_id):
        return "Still here, sir."

    ch._handler = handler
    _run(ch._poll_once())

    sends = [p for p in posts if p[0].endswith("/sendMessage")]
    assert sends == [(f"{ch.api}/sendMessage", {"chat_id": "555", "text": "Still here, sir."})]


# ------------------------------------------------------------------ regression

def test_default_channel_send_payload_unchanged():
    sent = {}

    def poster(url, json=None, timeout=None):
        sent["url"] = url
        sent["json"] = json
        return FakeResp(status_code=200)

    ch = TelegramChannel("TOK", poster=poster)
    assert ch.send("done, sir", "555") is True
    assert sent["json"] == {"chat_id": "555", "text": "done, sir"}
    assert ch.feedback_provider is None
    assert ch.on_callback is None
